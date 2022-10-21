import os, sys
import os.path as osp
import datetime
from utils.slconfig import SLConfig
from utils.vis_utils import BatchColorize
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from configs.myconfig import get_config
from datasets.get_loader import get_dataloader
from network.modelwarper import ModelWarper, ModelWarperV2
from regression.scopsp_regression import run_regression
from utils.utils import AverageMeter, FeiWu, Logger
from utils.plot_landmarks import plot_landmarks, unfold_heatmap

import matplotlib.pyplot as plt

train_step = 0

def main(args):
    args.gpu_number = len(args.gpu_devices.split(','))
    args.world_size = args.gpu_number * args.nodes

    if args.get('distributed', False):
        mp.spawn(run, nprocs=args.gpu_number, args=(args,))
        os.environ['MASTER_ADDR'] = args.MASTER_ADDR
        os.environ['MASTER_PORT'] = args.MASTER_PORT
    else:
        run(0, args)
    
def run(gpuid, args):
    """distribution init"""
    args.gpuid = gpuid
    if args.get('distributed', False):
        rank = gpuid 
        args.rank = rank
        args.gpuid = gpuid
        # rank: id of GPU for all nodes
        # gpuid: id of GPU this node

        # 1. init
        torch.distributed.init_process_group(
            backend="nccl",                                         
            init_method='env://',                                   
            world_size=args.world_size,                              
            rank=rank
        )
        # 2. config GPU for each process
        torch.cuda.set_device(gpuid)

    """CONFIG"""
    torch.backends.cudnn.benchmark = True
    # only have output on gpu:0
    if gpuid == 0:
        sys.stdout = Logger(osp.join(args.save_dir, 'train.log'))
        writer = SummaryWriter(osp.join(args.save_dir, 'writer'))
        os.makedirs(osp.join(args.save_dir, 'models'), exist_ok=True)
        os.makedirs(osp.join(args.save_dir, 'pics'), exist_ok=True)
    else:
        print("gpu=", gpuid, ' without output')
        sys.stdout = open(os.devnull, 'w') # No output
        writer = FeiWu() # No log output

    # some logs
    starttime = datetime.datetime.now()
    print("Start time: ", starttime, flush=True)
    print('command: python %s' % args.command, flush=True)
    print(args.pretty_text, flush=True)

    """DATA"""
    train_loader, test_loader, train_sampler = get_dataloader(args)
    print('len(train_set):', len(train_loader.dataset))

    """MODEL"""
    model = ModelWarperV2(args)
    # for k,v in model.named_modules():
    #     print(k)
    # raise ValueError
    model = model.cuda(gpuid)

    optimizer = Adam([{ 'params': model.parameters(),
                        'lr': args.TRAIN.lr}], weight_decay=args.TRAIN.weight_decay) 

    
    if args.get('distributed', False):
        model = nn.DataParallel(model, device_ids=[gpuid])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpuid], find_unused_parameters=True)
    else:
        model = nn.DataParallel(model)
        

    if args.get('SCHEDULER', False):
        print('Using %s' % args.SCHEDULER.name)
        scheduler = getattr(lr_scheduler, args.SCHEDULER.name)(optimizer, **args.SCHEDULER.paras)
    else:
        print('Using ReduceLROnPlateau')
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=2, verbose=True, min_lr=2e-6)

    """RESUME & PRETRAINED"""
    if args.get('resume', None) is not None:
        print("Loading checkpoint from '{}'".format(args.resume), flush=True)
        checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(gpuid))
        # load state dict to base model
        model.module.model.load_state_dict(checkpoint['state_dict'], strict=False)

    """RUN"""
    loss_best = 1e10
    for i in range(args.TRAIN.epochs):
        if args.TRAIN.get('freeze_epoch'):
            if (i+1 >= args.TRAIN.freeze_epoch):
                print('Switch To genarating mode!!!')
                optimizer = Adam([{ 'params': model.module.get_paras('gen_parameters'),
                        'lr': args.TRAIN.lr}], weight_decay=args.TRAIN.weight_decay) 
                args_new = args.copy()
                args_new.metriclist_train = [args_new.metriclist_train[0]]
                model.module.set_metrics(args_new)
                model.module = model.module.cuda(gpuid)
                
        if not args.get('skip_train', False):
            if args.get('distributed', False):
                train_sampler.set_epoch(i)
            loss_i = train(i, model, train_loader, optimizer, scheduler, gpuid, writer, args)

        # save model
        print('saving model')
        if args.gpuid == 0:
            state_dict = model.module.model.state_dict()
            saveinfo = {
                        'epoch': i + 1,
                        'state_dict': state_dict,
                    }
            torch.save(saveinfo, osp.join(args.save_dir, "models", "checkpoint.pkl"))

    endtime = datetime.datetime.now()
    print("Total time used: {}".format( endtime - starttime ), flush=True)


def train(epoch, model, train_loader, optimizer, scheduler, gpuid, writer, args):
    global train_step
    model.train()
    lossave = AverageMeter()
    lossave_all = {} # {name: Ave()}

    # lr_now = scheduler.get_lr()
    lr_now = [ group['lr'] for group in optimizer.param_groups ]
    writer.add_scalar('lr', lr_now[0], epoch + 1)
    len_train_loader = len(train_loader)

    # mixed_precision
    mixed_precision = args.get('mixed_precision', False)
    scaler = GradScaler()
    
    print(f'\n------------Epoch {epoch + 1} started!------------')
    print('=======> Train', flush=True)
    print('lr_now:', lr_now, flush=True)
    if args.gpuid == 0:
        batches = tqdm(train_loader, total=len_train_loader, position=0, leave=True, ascii=True)
    else:
        batches = train_loader

    for idx, sample in enumerate(batches):
        train_step += 1
        # get data
        img = sample['image'].cuda()

        ddata = {
            'img': img
        }

        # throw them into the model
        with autocast(enabled=mixed_precision):
            output_full = model(ddata, mode='pretrain_encoder')
            output = output_full['loss']

            # updata loss
            loss = output['loss'].mean()
            lossave.update(loss.item(), 1)
            for k,v in output['lossdict'].items():
                if k not in lossave_all:
                    lossave_all[k] = AverageMeter()
                # print(k,v)
                lossave_all[k].update(v.mean().item(), 1)

            # back prop
            optimizer.zero_grad()
            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        writer.add_scalar('Train/Loss', loss.item(), idx + epoch * len_train_loader + 1)
        writer.add_scalars('Train/Losses', {k:v.avg for k,v in lossave_all.items()}, idx + epoch * len_train_loader + 1)
        if args.gpuid == 0:
            batches.set_description('{} Loss: {:.6f}'.format(epoch + 1, lossave.avg))

        if idx % args.TRAIN.print_interval_train == 0:
            print("{} Loss: {:.6f}".format(epoch + 1, lossave.avg), flush=True)
            print({k:v.avg for k,v in lossave_all.items()}, flush=True)

        if args.testmode:
            if idx > 5:
                break

        if train_step % args.TRAIN.tb_plot_interval == 0:
            _h, _w = args.img_size
            # images
            if 'image' in sample:
                images = sample['image'].detach().cpu()[:8]
                writer.add_images('Train/image', images, train_step)

            # recovered img
            if 'img_rec' in output_full['output']:
                img_rec = output_full['output']['img_rec'].detach().cpu()[:8].numpy().astype(np.float32)
                writer.add_images('Train/img_rec', img_rec, train_step)

            # n color
            # n_color = hm_y_sm

    print(f'Epoch {epoch + 1} Loss: {lossave.avg}', flush=True)
    return lossave.avg
    

if __name__ == "__main__":
    args = get_config()
    main(args)