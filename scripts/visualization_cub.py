import os, sys
import os.path as osp

from torch.utils.data import dataloader
sys.path.append(osp.dirname(sys.path[0]))

from utils.soft_points import get_expected_points_from_map
from utils.vis_utils import BatchColorize
from tqdm import tqdm
import random

import cv2
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from configs.myconfig import get_config
from datasets.get_loader import get_dataloader
from network.modelwarper import ModelWarperV2
from regression.regression_sklearn import run_regression
from utils.utils import AverageMeter, FeiWu, Logger
from utils.plot_landmarks import plot_landmarks, unfold_heatmap
import imageio

import matplotlib.pyplot as plt

from utils.slconfig import SLConfig
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def main():
    args = args_model = SLConfig.fromfile('log/1103/SCOPSP_5class_cub-it1_percep-1.5x_clnew-1.5x_bg-0.3x_fd-0.1x_fo-1.0x_32x32_cont/config-SCOPSP_CUB_BG_Norm_128x128_TPS_PERCEPNEW_10lm_hm32_newtc copy.py')
    args_data = args # SLConfig.fromfile('')
    args.resume = 'log/1103/SCOPSP_5class_cub-it1_percep-1.5x_clnew-1.5x_bg-0.3x_fd-0.1x_fo-1.0x_32x32_cont/models/ep30.pkl'
    output_dir = 'vis/cub_extent'
    args.save_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # config log    
    """LOGGER"""
    sys.stdout = Logger(osp.join(args.save_dir, 'test.log'))
    os.makedirs(osp.join(args.save_dir, 'models'), exist_ok=True)
    os.makedirs(osp.join(args.save_dir, 'pics'), exist_ok=True)

    """DATA"""
    # args_data.DATASET.TRAIN_SET.paras.pic_trans_num = 0
    args_data.DATASET.TRAIN_SET.shuffle = False
    train_loader, test_loader, train_sampler = get_dataloader(args_data)
    print('len(train_set):', len(train_loader.dataset))
    print('len(test_set):', len(test_loader.dataset))

    """MODEL"""
    model = ModelWarperV2(args)
    # for k,v in model.named_modules():
    #     print(k)
    # raise ValueError
    model = model.cuda()
    model = nn.DataParallel(model)
    _ = model.eval()

    """RESUME & PRETRAINED"""
    if args.get('resume', None) is not None:
        print("Loading checkpoint from '{}'".format(args.resume), flush=True)
        checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda())
        # load state dict to base model
        model.module.model.load_state_dict(checkpoint['state_dict'], strict=False)

    dataloaderdict = {
        'train': train_loader,
        'test': test_loader,
    }
    plot_img(model, train_loader, args)

def save_img(model, image, imgnames, colorizer, args):
    rawimg_dir = osp.join(args.save_dir, 'rawimg')
    segmap_dir = osp.join(args.save_dir, 'segmap')
    overlay_dir = osp.join(args.save_dir, 'overlay')

    ddata = {
        'img': image,
    }
    output = model(ddata, mode='get_hm_pred', get_loss=False)['output']
    hm_sm = output['hm_sm'].detach().cpu()
    hm_sm = nn.functional.upsample(hm_sm, (128, 128), mode='bilinear', align_corners=True)
    hm_sm = hm_sm.argmax(1)
    seg_vis = colorizer(hm_sm.numpy()).transpose(0, 2, 3, 1)
    # print('seg_vis.shape:', seg_vis.shape)
    # raise ValueError

    img_raw = image.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img_overlay = (seg_vis * 0.7 + img_raw * 0.8).clip(0,1)

    B = image.shape[0]
    for i in range(B):
        imgname = imgnames[i]
        imgname = imgname.replace('/', '_')
        rawimg = img_raw[i]
        # print('rawimg.shape:',  rawimg.shape)
        imageio.imwrite(osp.join(rawimg_dir, imgname+'.raw.png'), rawimg)
        segmap = seg_vis[i]
        imageio.imwrite(osp.join(segmap_dir, imgname+'.segmap.png'), segmap)
        overlay = img_overlay[i]
        imageio.imwrite(osp.join(overlay_dir, imgname+'.overlay.png'), overlay)

def plot_img(model, dataloader, args):
    _ = model.eval()
    batches = tqdm(dataloader, total=len(dataloader), position=0, leave=True, ascii=True)
    colorizer = BatchColorize(n=args.lm_numb)
    rawimg_dir = osp.join(args.save_dir, 'rawimg')
    segmap_dir = osp.join(args.save_dir, 'segmap')
    overlay_dir = osp.join(args.save_dir, 'overlay')
    os.makedirs(rawimg_dir, exist_ok=True)
    os.makedirs(segmap_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    for idx, sample in enumerate(tqdm(batches)):
        if idx > 50:
            break
        # get data
        image = sample['image'].cuda()
        imgnames = sample['imgname']
        save_img(model, image, imgnames, colorizer, args)

        image = sample['img1']['image_trans'].cuda()
        imgnames = [x+'trans1' for x in sample['imgname']]
        save_img(model, image, imgnames, colorizer, args)

        image = sample['img2']['image_trans'].cuda()
        imgnames = [x+'trans2' for x in sample['imgname']]
        save_img(model, image, imgnames, colorizer, args)



        

if __name__ == "__main__":
    main()

