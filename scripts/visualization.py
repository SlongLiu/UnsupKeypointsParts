import os, sys
import os.path as osp
from PIL import Image

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
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def main():
    args = args_model = SLConfig.fromfile('log/1112/SCOPSP_5class_VOC-areo_percep-1.5x_clnew-3.0x_bg-0.3x_arc-0.1x_32x32_SOTA/config-SCOPSP_VOC_BG_Norm_128x128_TPS_PERCEPNEW_ARC_10lm_hm32_newtc.py')
    args_data = args # SLConfig.fromfile('')
    args.resume = 'log/1112/SCOPSP_5class_VOC-areo_percep-1.5x_clnew-3.0x_bg-0.3x_arc-0.1x_32x32_SOTA/models/ep30.pkl'
    output_dir = 'vis/voc-aero2'
    args.save_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # config log    
    """LOGGER"""
    sys.stdout = Logger(osp.join(args.save_dir, 'test.log'))
    os.makedirs(osp.join(args.save_dir, 'models'), exist_ok=True)
    os.makedirs(osp.join(args.save_dir, 'pics'), exist_ok=True)

    """DATA"""
    args_data.DATASET.TRAIN_SET.paras.pic_trans_num = 0
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
        if 'bbox_used' in sample:
            # raw_images = sample['image_raw'].numpy()
            bbox_used = sample['bbox_used'].numpy()
            raw_img_sizes = sample['raw_img_size'].numpy()
            imgpaths = sample['imgpath']

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

            # if 'bbox_used' in sample:
            #     bbox_used_i = bbox_used[i]
            #     img_size = img_sizes[i]
            #     tw, th = int(bbox_used_i[2]-bbox_used_i[0]), int(bbox_used_i[3]-bbox_used_i[1])
            #     # print("bbox_used_i：", bbox_used_i)
            #     seg_vis_resize = seg_vis[i].copy() * 255
            #     seg_vis_resize = cv2.resize(seg_vis_resize, (tw, th), interpolation=cv2.INTER_CUBIC)

            #     raw_image_pil = Image.open(imgpaths[i]).convert("RGB")
            #     # print(seg_vis_resize.dtype)
            #     # print('seg_vis_resize.shape:', seg_vis_resize.shape)
            #     seg_vis_pil = Image.fromarray(seg_vis_resize.astype(np.uint8)).convert("RGB")
            #     if bbox_used_i[0] < 0:
            #         seg_vis_pil = seg_vis_pil.crop(np.maximum(-bbox_used_i, 0))

            #     # print('raw_image_pil:', raw_image_pil, "\nseg_vis_pil", seg_vis_pil, "\n", raw_image_pil.size, seg_vis_pil.size)

            #     raw_image_pil.paste(seg_vis_pil, (max(int(bbox_used_i[0]),0), max(int(bbox_used_i[1]),0)))
            #     # print(img_size[0], img_size[1])
            #     raw_image_pil = raw_image_pil.crop((0, 0, img_size[0], img_size[1]))

            #     os.makedirs(osp.join(osp.dirname(rawimg_dir), 'origin'), exist_ok=True)
            #     imageio.imwrite(osp.join(osp.dirname(rawimg_dir), 'origin', imgname+'.origin.png'), np.array(raw_image_pil))

            imageio.imwrite(osp.join(rawimg_dir, imgname+'.raw.png'), (rawimg*255).astype(np.uint8))
            segmap = seg_vis[i]
            imageio.imwrite(osp.join(segmap_dir, imgname+'.segmap.png'), (segmap*255).astype(np.uint8))
            overlay = img_overlay[i]
            imageio.imwrite(osp.join(overlay_dir, imgname+'.overlay.png'), (overlay*255).astype(np.uint8))
        
            if 'bbox_used' in sample:
                bbox_used_i = bbox_used[i]
                raw_img_size = raw_img_sizes[i]
                w, h = raw_img_size
                tw, th = int(bbox_used_i[2]-bbox_used_i[0]), int(bbox_used_i[3]-bbox_used_i[1])

                # v2
                segmap = segmap * 255
                segmap = cv2.resize(segmap, (tw, th), interpolation=cv2.INTER_CUBIC)
                seg_pil = Image.fromarray(segmap.astype(np.uint8)).convert("RGB")
                seg_pil = seg_pil.crop((max(0, -bbox_used_i[0]), max(0, -bbox_used_i[1]), min(tw, w-bbox_used_i[0]), min(th, h-bbox_used_i[1])))

                bg_0 = np.zeros((h, w, 3), dtype=np.uint8)
                bg_pil = Image.fromarray(bg_0).convert("RGB")
                bg_pil.paste(seg_pil, (max(int(bbox_used_i[0]),0), max(int(bbox_used_i[1]),0)))
                seg_res = np.array(bg_pil)

                # save
                raw_image_full = np.array(Image.open(imgpaths[i]).convert("RGB"))
                overlay_v4 = (seg_res * 0.7 + raw_image_full * 0.8).astype(np.uint8)
                os.makedirs(osp.join(osp.dirname(rawimg_dir), 'overlay_v4'), exist_ok=True)
                imageio.imwrite(osp.join(osp.dirname(rawimg_dir), 'overlay_v4', imgname+'.overlay_v4.png'), np.array(overlay_v4))

                seg_res = cv2.resize(seg_res, (128, 128), interpolation=cv2.INTER_CUBIC)
                raw_image_full = cv2.resize(raw_image_full, (128, 128), interpolation=cv2.INTER_CUBIC)
                overlay_v4_resize = (seg_res * 0.6 + raw_image_full * 0.7).astype(np.uint8)
                os.makedirs(osp.join(osp.dirname(rawimg_dir), 'overlay_v5'), exist_ok=True)
                imageio.imwrite(osp.join(osp.dirname(rawimg_dir), 'overlay_v5', imgname+'.overlay_v5.png'), np.array(overlay_v4_resize))

                # overlay = overlay * 255
                # overlay_resize = cv2.resize(overlay, (tw, th), interpolation=cv2.INTER_CUBIC)

                # raw_image_pil = Image.open(imgpaths[i]).convert("RGB")
                # os.makedirs(osp.join(osp.dirname(rawimg_dir), 'pil'), exist_ok=True)
                # imageio.imwrite(osp.join(osp.dirname(rawimg_dir), 'pil', imgname+'.pil.png'), np.array(raw_image_pil))


                # raw_image_pil = Image.fromarray((np.array(raw_image_pil) * 0.8).astype(np.uint8)).convert("RGB")
                # # print(seg_vis_resize.dtype)
                # # print('seg_vis_resize.shape:', seg_vis_resize.shape)
                # overlay_pil = Image.fromarray(overlay_resize.astype(np.uint8)).convert("RGB")
                # overlay_pil = overlay_pil.crop((max(0, -bbox_used_i[0]), max(0, -bbox_used_i[1]), min(tw, w-bbox_used_i[0]), min(th, h-bbox_used_i[1])))
                # # print(overlay_pil.size)

                # os.makedirs(osp.join(osp.dirname(rawimg_dir), 'overlay_pil'), exist_ok=True)
                # imageio.imwrite(osp.join(osp.dirname(rawimg_dir), 'overlay_pil', imgname+'.overlay_pil.png'), np.array(overlay_pil))

                # # print('raw_image_pil:', raw_image_pil, "\nseg_vis_pil", seg_vis_pil, "\n", raw_image_pil.size, seg_vis_pil.size)

                # raw_image_pil.paste(overlay_pil, (max(int(bbox_used_i[0]),0), max(int(bbox_used_i[1]),0)))
                # raw_image_pil = raw_image_pil.crop((0, 0, raw_img_size[0], raw_img_size[1]))

                # os.makedirs(osp.join(osp.dirname(rawimg_dir), 'overlay2'), exist_ok=True)
                # imageio.imwrite(osp.join(osp.dirname(rawimg_dir), 'overlay2', imgname+'.overlay2.png'), np.array(raw_image_pil))

                # raw_image_pil = raw_image_pil.resize((128, 128))
                # os.makedirs(osp.join(osp.dirname(rawimg_dir), 'overlay3'), exist_ok=True)
                # imageio.imwrite(osp.join(osp.dirname(rawimg_dir), 'overlay3', imgname+'.overlay3.png'), np.array(raw_image_pil))


if __name__ == "__main__":
    main()

