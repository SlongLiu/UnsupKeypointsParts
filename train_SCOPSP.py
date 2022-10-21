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
            # scheduler.step(loss_i)

        # if not args.get('skip_plot', False) and i % args.TRAIN.plot_interval == 0 \
        #     and args.gpuid == 0:
        #     datasetdict = {
        #         'train': train_loader.dataset,
        #         'test': test_loader.dataset,
        #         'hard': train_loader.dataset
        #     }
        #     plot_landmarks_epoch(i, model, datasetdict, args)

        reg_list = args.get('reg_list', None)
        if reg_list is not None:
            if isinstance(reg_list, str):
                reg_list = [reg_list]
            for reg_name in reg_list:
                cfgfile = f'configs/SCOPSP/TEST_{reg_name}_SCOPS_DATASET.py'
                args_data = SLConfig.fromfile(cfgfile)
                reg_train_loader, reg_test_loader, train_sampler = get_dataloader(args_data)
                dataloaderdict = {
                    'train': reg_train_loader,
                    'test': reg_test_loader,
                }
                mean_error = run_regression(model, dataloaderdict, writer, args, dtype=reg_name)
                print(f'{reg_name}, ep{i+1}: {mean_error}')
                writer.add_scalar(f'celeba_mean/{reg_name}', mean_error, i+1)                

        if args.DATASET.TRAIN_SET.dataset == 'VOC':
            from regression.iou_calcu import calcu_iou
            iou = calcu_iou(model, test_loader, writer, args, args.classselect)
            writer.add_scalar(f'VOC/{args.classselect}', iou, i+1)

        # if args.get('regression_celeba', False):
        #     args_data = SLConfig.fromfile('configs/SCOPSP/TEST_wildceleba_SCOPS_DATASET.py')
        #     reg_train_loader, reg_test_loader, train_sampler = get_dataloader(args_data)
        #     dataloaderdict = {
        #         'train': reg_train_loader,
        #         'test': reg_test_loader,
        #     }
        #     mean_error = run_regression(model, dataloaderdict, writer, args)
        #     print(f'celeba, ep{i+1}: {mean_error}')
        #     writer.add_scalar('celeba_mean', mean_error, i+1)

        # if args.get('regression_AFLW', False):
        #     args_data = SLConfig.fromfile('configs/SCOPSP/TEST_wildAFLW_SCOPS_DATASET.py')
        #     reg_train_loader, reg_test_loader, train_sampler = get_dataloader(args_data)
        #     dataloaderdict = {
        #         'train': reg_train_loader,
        #         'test': reg_test_loader,
        #     }
        #     mean_error = run_regression(model, dataloaderdict, writer, args)
        #     print(f'AFLW, ep{i+1}: {mean_error}')
        #     writer.add_scalar('celeba_mean', mean_error, i+1)

    endtime = datetime.datetime.now()
    print("Total time used: {}".format( endtime - starttime ), flush=True)


def train(epoch, model, train_loader, optimizer, scheduler, gpuid, writer, args):
    global train_step
    model.train()
    lossave = AverageMeter()
    lossave_all = {} # {name: Ave()}
    colorizer = BatchColorize(n=args.lm_numb)

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
        img1 = sample['img1']['image_trans'].cuda()
        img1_mask = sample['img1']['image_mask'].cuda()
        img2 = sample['img2']['image_trans'].cuda()
        img2_mask = sample['img2']['image_mask'].cuda()

        ddata = {
            'img1': img1,
            'img2': img2,
            'img1_mask': img1_mask,
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
                images = sample['image'][:8]
                writer.add_images('Train/images', images, train_step)
            img1_vis = img1.detach().cpu()[:8]
            writer.add_images('Train/img1', img1_vis, train_step)
            img2_vis = img2.detach().cpu()[:8]
            writer.add_images('Train/img2', img2_vis, train_step)

            # heatmaps
            hm_x_sm = output_full['output']['hm_x_sm'].detach().cpu()[:8] # B,10,16,16
            hm_x_sm = nn.functional.upsample(hm_x_sm, (_h, _w), mode='bilinear', align_corners=True)
            hm_x_sm = hm_x_sm.argmax(1) # B,128,128
            seg_vis1 = colorizer(hm_x_sm.numpy())
            writer.add_images('Train/img1_seg', seg_vis1, train_step)
            # print(seg_vis1)
            img1_all = (img1_vis.numpy() * 0.5 + seg_vis1 * 0.5)
            writer.add_images('Train/img1_all', img1_all, train_step)

            hm_y_sm = output_full['output']['hm_y_sm'].detach().cpu()[:8]
            hm_y_sm = nn.functional.upsample(hm_y_sm, (_h, _w), mode='bilinear', align_corners=True)
            hm_y_sm = hm_y_sm.argmax(1) # B,128,128
            seg_vis2 = colorizer(hm_y_sm.numpy())
            writer.add_images('Train/img2_seg', seg_vis2, train_step)
            img2_all = (img2_vis.numpy() * 0.5 + seg_vis2 * 0.5)
            writer.add_images('Train/img2_all', img2_all, train_step)

            # recovered img
            if 'recovered_x' in output_full['output']:
                recovered_img1 = output_full['output']['recovered_x'].detach().cpu()[:8].numpy().astype(np.float32)
                writer.add_images('Train/recovered_img1', recovered_img1, train_step)

            if 'recovered_y' in output_full['output']:
                recovered_img2 = output_full['output']['recovered_y'].detach().cpu()[:8].numpy().astype(np.float32)
                writer.add_images('Train/recovered_img2', recovered_img2, train_step)

            # mask
            mask1 = img1_mask.detach().cpu()[:8]
            writer.add_images('Train/mask1', mask1, train_step)
            mask2 = img2_mask.detach().cpu()[:8]
            writer.add_images('Train/mask2', mask2, train_step)

            # n color
            # n_color = hm_y_sm

    print(f'Epoch {epoch + 1} Loss: {lossave.avg}', flush=True)
    

def test(epoch, model, test_loader, gpuid, writer, args):
    lossave = AverageMeter()
    lossave_all = {} # {name: Ave()}
    mixed_precision = args.get('mixed_precision', False)
    colorizer = BatchColorize(n=args.lm_numb)

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
            img1_mask = sample['img1']['image_mask'].cuda()
            img2_mask = sample['img2']['image_mask'].cuda()

            ddata = {
                'img1': img1,
                'img2': img2,
                'img1_mask': img1_mask,
                'img2_mask': img2_mask,
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

    # plot:
    if 1:
        # images
        _h, _w = args.img_size
        if 'image' in sample:
            images = sample['image'][:8]
            writer.add_images('Test/images', images, epoch + 1)
        img1_vis = img1.detach().cpu()[:8]
        writer.add_images('Test/img1', img1_vis, epoch + 1)
        img2_vis = img2.detach().cpu()[:8]
        writer.add_images('Test/img2', img2_vis, epoch + 1)

        # heatmaps
        hm_x_sm = output_full['output']['hm_x_sm'].detach().cpu()[:8] # B,10,16,16
        hm_x_sm = nn.functional.upsample(hm_x_sm, (_h, _w), mode='bilinear', align_corners=True)
        hm_x_sm = hm_x_sm.argmax(1) # B,128,128
        seg_vis1 = colorizer(hm_x_sm.numpy())
        writer.add_images('Test/img1_seg', seg_vis1, epoch + 1)
        
        # overlay img
        img1_all = (img1_vis.numpy() * 0.5 + seg_vis1 * 0.5)
        writer.add_images('Test/img1_all', img1_all, epoch + 1)
        img1_colorful = (img1_vis.numpy() * 0.8 + seg_vis1 * 0.7).clip(0,1.0)
        writer.add_images('Test/img1_colorful', img1_colorful, epoch + 1)

        hm_y_sm = output_full['output']['hm_y_sm'].detach().cpu()[:8]
        hm_y_sm = nn.functional.upsample(hm_y_sm, (_h, _w), mode='bilinear', align_corners=True)
        hm_y_sm = hm_y_sm.argmax(1) # B,128,128
        seg_vis2 = colorizer(hm_y_sm.numpy())
        writer.add_images('Test/img2_seg', seg_vis2, epoch + 1)
        
        # overlay img
        img2_all = (img2_vis.numpy() * 0.5 + seg_vis2 * 0.5)
        writer.add_images('Test/img2_all', img2_all, epoch + 1)
        img2_colorful = (img2_vis.numpy() * 0.8 + seg_vis2 * 0.7).clip(0,1.0)
        writer.add_images('Test/img2_colorful', img2_colorful, epoch + 1)

        # recovered img
        if 'recovered_x' in output_full['output']:
            recovered_img1 = output_full['output']['recovered_x'].detach().cpu()[:8].numpy().astype(np.float32)
            writer.add_images('Test/recovered_img1', recovered_img1, epoch + 1)

        if 'recovered_y' in output_full['output']:
            recovered_img2 = output_full['output']['recovered_y'].detach().cpu()[:8].numpy().astype(np.float32)
            writer.add_images('Test/recovered_img2', recovered_img2, epoch + 1)

        # mask
        mask1 = img1_mask.detach().cpu()[:8]
        writer.add_images('Test/mask1', mask1, epoch + 1)
        mask2 = img2_mask.detach().cpu()[:8]
        writer.add_images('Test/mask2', mask2, epoch + 1)

    # save the model
    if not args.TRAIN.get('only_save_best_model', False):
        print('saving model')
        if args.gpuid == 0:
            state_dict = model.module.model.state_dict()
            saveinfo = {
                        'epoch': epoch + 1,
                        'state_dict': state_dict,
                        'loss': lossave.avg,
                    }
            torch.save(saveinfo, osp.join(args.save_dir, "models", "ep%d.pkl" % (epoch+1)))
    return lossave.avg







if __name__ == "__main__":
    args = get_config()
    main(args)