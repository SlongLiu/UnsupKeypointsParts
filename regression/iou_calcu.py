import os, sys
import os.path as osp
from PIL import Image


from torch.utils.tensorboard import writer
if __name__ == "__main__":
    sys.path.append(osp.dirname(sys.path[0]))
import datetime

from sklearn import linear_model
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
from network.modelwarper import ModelWarper, ModelWarperV2
from regression.regression_sklearn import run_regression
from utils.utils import AverageMeter, FeiWu, Logger
from utils.plot_landmarks import plot_landmarks, unfold_heatmap

import matplotlib.pyplot as plt

from utils.slconfig import SLConfig

# CUDA_VISIBLE_DEVICES 

def main():
    args = args_model = SLConfig.fromfile('log/1104/SCOPSP_5class_VOC-sheep_percep-1.0x_clnew-1.0x_bg-0.3x_arc+-1.0x_32x32/config-SCOPSP_VOC_BG_Norm_128x128_TPS_PERCEPNEW_ARC_10lm_hm32_newtc.py')
    args.resume = 'log/1104/SCOPSP_5class_VOC-sheep_percep-1.0x_clnew-1.0x_bg-0.3x_arc+-1.0x_32x32/models/ep60.pkl'
    args.classselect = 'sheep'

    # config log    
    """LOGGER"""
    sys.stdout = Logger(osp.join(args.save_dir, 'test.log'))
    writer = SummaryWriter(osp.join(args.save_dir, 'writer'))
    os.makedirs(osp.join(args.save_dir, 'models'), exist_ok=True)
    os.makedirs(osp.join(args.save_dir, 'pics'), exist_ok=True)

    """DATA"""
    train_loader, test_loader, train_sampler = get_dataloader(args)
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

    test_loader.dataset.pic_trans_num = 0
    calcu_iou(model, test_loader, writer, args, classtype=args.classselect)
    test_loader.dataset.pic_trans_num = 2

def calcu_iou(model, dataloader, writer, args, classtype):
    classmap = dict(aeroplane=1, bicycle=2, bird=3, boat=4, bottle=5, bus=6, car=7 , cat=8, chair=9, cow=10, diningtable=11, dog=12, horse=13, motorbike=14, person=15, potted_plant=16, sheep=17, sofa=18, train=19, tv_monitor=20)
    assert classtype in classmap
    # init
    _ = model.eval()
    batches = tqdm(dataloader, total=len(dataloader), position=0, leave=True, ascii=True)
    iouAM = AverageMeter()

    # gen json lm
    for idx, sample in enumerate(tqdm(batches)):
        # get data
        image = sample['image'].cuda()
        segmappath = sample['segmappath'] # B,3,H,W

        ddata = {
            'img': image,
        }
        output = model(ddata, mode='get_hm_pred', get_loss=False)['output']
        hm_sm = output['hm_sm'].detach().cpu().numpy() # B,N,H,W
        hm_sm = hm_sm.transpose(0,2,3,1)
        # pred_segmaps = 1 - (hm_sm.argmax(1) == 0) # B,H',W'
        for i in range(hm_sm.shape[0]):
            # gt_segmap = cv2.imread(segmappath[i])[:,:,0] # h,w
            gt_segmap = Image.open(segmappath[i])
            if 'bbox_used' in sample:
                gt_segmap = gt_segmap.crop(sample['bbox_used'][i].numpy())
            gt_segmap = np.array(gt_segmap) # h,w
            if classtype == 'motorbike':
                gt_segmap = (gt_segmap == classmap[classtype]) | (gt_segmap == 15) 
            else:
                gt_segmap = (gt_segmap == classmap[classtype])
            h0, w0 = gt_segmap.shape

            pred_segmap = cv2.resize(hm_sm[i], (w0, h0), interpolation=cv2.INTER_CUBIC)
            pred_segmap = (pred_segmap.argmax(-1) != 0)
            assert gt_segmap.shape == pred_segmap.shape, f'{gt_segmap.shape} != {pred_segmap.shape}'

            iou = (gt_segmap & pred_segmap).sum() / (gt_segmap | pred_segmap).sum()
            iouAM.update(iou)
    print(f"iou of {classtype}: {iouAM.avg}")
    return iouAM.avg

if __name__ == "__main__":
    main()
