import os, sys
import os.path as osp
import datetime
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from torch.cuda.amp import GradScaler, autocast

from configs.myconfig import get_config
from datasets.get_loader import get_dataloader
from network.modelwarper import ModelWarper, ModelWarperV2
from regression.regression_sklearn import run_regression
from utils.utils import AverageMeter, FeiWu, Logger
from utils.plot_landmarks import plot_landmarks, unfold_heatmap

import matplotlib.pyplot as plt


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
    print('len(test_set):', len(test_loader.dataset))

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
    if args.first_test:
        _ = test(-1, model, test_loader, gpuid, writer, args)

    loss_best = 1e10
    for i in range(args.TRAIN.epochs):
        if not args.get('skip_train', False):
            if args.get('distributed', False):
                train_sampler.set_epoch(i)
            train(i, model, train_loader, optimizer, scheduler, gpuid, writer, args)

        if not args.get('skip_test', False) and i % args.TRAIN.test_interval == 0:
            loss_i = test(i, model, test_loader, gpuid, writer, args)
            if loss_i < loss_best:
                loss_best = loss_i
                # save the model
                if args.gpuid == 0:
                    state_dict = model.module.model.state_dict()
                    saveinfo = {
                                'epoch': i + 1,
                                'state_dict': state_dict,
                                'loss': loss_i,
                            }
                    torch.save(saveinfo, osp.join(args.save_dir, "models", "test_best.pkl"))
            scheduler.step(loss_i)

        if not args.get('skip_plot', False) and i % args.TRAIN.plot_interval == 0 \
            and args.gpuid == 0:
            datasetdict = {
                'train': train_loader.dataset,
                'test': test_loader.dataset,
                'hard': train_loader.dataset
            }
            plot_landmarks_epoch(i, model, datasetdict, args)

        if not args.get('skip_regression', False) and \
        (i + 1 == args.TRAIN.epochs or \
        (i+1) % args.TRAIN.get('regression_interval', 100000) == 0):
            
            if 'REGRESSION_MAFL' in args:
                # argsnew = args.deepcopy()
                # argsnew.save_dir = osp.join(argsnew.save_dir, "reg_ep%d"%(i+1))
                # os.makedirs(argsnew.save_dir, exist_ok=True)
                print()
                print("******MAFL******")
                rg_train_loader, rg_test_loader, rg_train_sampler = get_dataloader(args.REGRESSION_MAFL)
                dataloaderdict = {
                    'train': rg_train_loader,
                    'test': rg_test_loader
                }
                run_regression(i, model, dataloaderdict, writer, args.REGRESSION_MAFL)

            if 'REGRESSION_AFLW' in args:
                print()
                print("******AFLW******")
                rg_train_loader, rg_test_loader, rg_train_sampler = get_dataloader(args.REGRESSION_AFLW)
                dataloaderdict = {
                    'train': rg_train_loader,
                    'test': rg_test_loader,
                }
                run_regression(i, model, dataloaderdict, writer, args.REGRESSION_AFLW)
                print()

    endtime = datetime.datetime.now()
    print("Total time used: {}".format( endtime - starttime ), flush=True)


def train(epoch, model, train_loader, optimizer, scheduler, gpuid, writer, args):
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
        # get data
        img1 = sample['img1']['image_trans'].cuda()
        img2 = sample['img2']['image_trans'].cuda()
        img2_mask = sample['img2']['image_mask'].cuda()

        ddata = {
            'img1': img1,
            'img2': img2,
            'img2_mask': img2_mask
        }

        # throw them into the model
        with autocast(enabled=mixed_precision):
            output_full = model(ddata)
            output = output_full['loss']

            # updata loss
            loss = output['loss'].mean()
            lossave.update(loss.item(), 1)
            for k,v in output['lossdict'].items():
                if k not in lossave_all:
                    lossave_all[k] = AverageMeter()
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

    print(f'Epoch {epoch + 1} Loss: {lossave.avg}', flush=True)
    


def test(epoch, model, test_loader, gpuid, writer, args):
    lossave = AverageMeter()
    lossave_all = {} # {name: Ave()}
    mixed_precision = args.get('mixed_precision', False)

    _ = model.eval()
    print('=======> Test', flush=True)
    with torch.no_grad():
        if args.gpuid == 0:
            batches = tqdm(test_loader, total=len(test_loader), position=0, leave=True, ascii=True)
        else:
            batches = test_loader
        for idx, sample in enumerate(batches):

            # get data
            img1 = sample['img1']['image_trans'].cuda()
            img2 = sample['img2']['image_trans'].cuda()
            img2_mask = sample['img2']['image_mask']

            ddata = {
                'img1': img1,
                'img2': img2,
                'img2_mask': img2_mask
            }

            # throw them into the model
            with autocast(enabled=mixed_precision):
                output_full = model(ddata)
                output = output_full['loss']

                # updata loss
                loss = output['loss'].mean()
                lossave.update(loss.item(), 1)
                for k,v in output['lossdict'].items():
                    if k not in lossave_all:
                        lossave_all[k] = AverageMeter()
                    lossave_all[k].update(v.mean().item(), 1)

            if args.gpuid == 0:
                batches.set_description('{} Test Loss: {:.6f}'.format(epoch + 1, lossave.avg))
            if args.testmode:
                if idx > 5:
                    break
    
    print(f'Epoch {epoch + 1} Loss: {lossave.avg}', flush=True)
    print({k:v.avg for k,v in lossave_all.items()}, flush=True)

    writer.add_scalar('Test/Loss', lossave.avg, epoch+1)
    writer.add_scalars('Test/Losses', {
        k: v.avg for k,v in lossave_all.items()
    }, epoch+1)

    # save the model
    if not args.TRAIN.get('only_save_best_model', False):
        if args.gpuid == 0:
            state_dict = model.module.model.state_dict()
            saveinfo = {
                        'epoch': epoch + 1,
                        'state_dict': state_dict,
                        'loss': lossave.avg,
                    }
            torch.save(saveinfo, osp.join(args.save_dir, "models", "ep%d.pkl" % (epoch+1)))
    return lossave.avg

import numpy as np

def plot_landmarks_epoch(epoch, model, datasetdict, args):
    _ = model.eval()
    print('=======> Plot', flush=True)
    for dname, dataset in datasetdict.items():
        n = 4
        col = 7
        row = 12
        plt.figure(figsize=(3*col, 3*row), dpi=80) # 12 x 7

        if dname == 'train':
            idxs = random.sample(list(range(len(dataset))), n)
        elif dname == 'test':
            idxs= list(range(n))
        elif dname == 'hard':
            idxs= [2, 571, 830, 1337] #
        else:
            raise NotImplementedError

        for cnt, idx in enumerate(idxs):
            # get image and run model
            with torch.no_grad():
                # data get
                sample = dataset[idx]
                
                # get data
                img1 = sample['img1']['image_trans'].unsqueeze(0).cuda()
                img2 = sample['img2']['image_trans'].unsqueeze(0).cuda()
                img2_mask = sample['img2']['image_mask'].unsqueeze(0)
                image = sample['image'].unsqueeze(0)
                lm_gt = sample['landmarks'] * args.img_size
                img2_mask_plot = sample['img2']['image_mask'].squeeze(0)
                img_name = sample['imgname']

                # throw them into the model
                ddata = {
                    'img1': img1,
                    'img2': img2,
                    'img2_mask': img2_mask
                }

                # throw them into the model
                output_full = model(ddata)
                lossdict = {k:v.mean().item() for k,v in output_full['loss']['lossdict'].items()}
                output = output_full['output']
                lm_y = output['lm_y'].detach().cpu().numpy()[0] * args.img_size
                if 'pb_y' in output:
                    pb_y = output['pb_y'][0,:,0,0].detach().cpu().numpy()
                else:
                    pb_y = None

                # denormalize
                mean = torch.Tensor(args.normalize_mean)
                std = torch.Tensor(args.normalize_std)
                
            ## plot the imgs
            ax1 = plt.subplot(row, col, cnt*3*col + 1)
            img1 = (img1.detach().cpu().squeeze(0).permute(1,2,0) * std) + mean
            ax1.imshow(img1)
            text = '\n'.join(["%s:%.6f" % (k,v) for k,v in lossdict.items()])
            ax1.set_title(text)
            ax1.set_axis_off()
            
            ax2 = plt.subplot(row, col, cnt*3*col + 2)
            img2 = (img2.detach().cpu().squeeze(0).permute(1,2,0) * std) + mean
            ax2.imshow(img2)
            ax2.set_axis_off()
            ax2.set_title(img_name)

            ax3 = plt.subplot(row, col, cnt*3*col + 3)
            img2_revocer = (output['recovered_y'].detach().cpu().squeeze(0).permute(1,2,0) * std) + mean
            ax3.imshow(img2_revocer)
            ax3.set_title("recovered_y")
            ax3.set_axis_off()

            ax4 = plt.subplot(row, col, cnt*3*col + 4)
            image = (image.detach().cpu().squeeze(0).permute(1,2,0) * std) + mean
            ax4.imshow(image)
            # ax4.set_title("raw image")
            ax4.set_axis_off()
            plot_landmarks(ax4, lm_gt)

            ax5 = plt.subplot(row, col, cnt*3*col + 5)
            ax5.imshow(img2)
            # ax4.set_title("%d: pred" % idx)
            ax5.set_axis_off()
            plot_landmarks(ax5, lm_y)

            ax6 = plt.subplot(row, col, cnt*3*col + 6)
            ax6.imshow(img2_mask_plot, cmap='gray')
            # ax4.set_title("%d: pred" % idx)
            ax6.set_axis_off()

            ax7 = plt.subplot(row, col, cnt*3*col + 7)
            ax7.imshow(img2_mask_plot[:, :, np.newaxis]*img2)
            # ax4.set_title("%d: pred" % idx)
            ax7.set_axis_off()

            ## plot heatmap
            axhm1 = plt.subplot2grid((row, col), (3*cnt + 1, 0), colspan=col)
            pose_y = unfold_heatmap(output['hmr_y'])
            axhm1.imshow(pose_y, cmap='gray')
            axhm1.set_axis_off()

            axhm2 = plt.subplot2grid((row, col), (3*cnt + 2, 0), colspan=col)
            pose_raw_y = unfold_heatmap(output['hm_y'])
            axhm2.imshow(pose_raw_y, cmap='gray')
            axhm2.set_axis_off()
            if pb_y is not None:
                axhm2.set_title(' '.join(["%0.6f" % ii for ii in pb_y.tolist()]))
        
        # savefig
        savefilename = osp.join(args.save_dir, 'pics/ep%s-%s.png' % (str(epoch+1), dname) )
        os.makedirs(osp.dirname(savefilename), exist_ok=True)
        plt.savefig(savefilename)
        plt.close()



if __name__ == "__main__":
    args = get_config()
    main(args)