import os, sys
import os.path as osp
from PIL import Image

from torch._C import dtype
from torch.utils.tensorboard import writer
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
    # args = args_model = SLConfig.fromfile('log/1027/SCOPSP_9class_wildceleba_percep-1.0x_clnew-1.5x_bg-0.3x_fd-0.1x_fo-0.1x_32x32_2/config-SCOPSP_WILDCELEBA_BG_Norm_128x128_TPS_PERCEPNEW_10lm_hm32_newtc.py')
    # args_data = SLConfig.fromfile('configs/SCOPSP/TEST_wildAFLW_SCOPS_DATASET.py')
    # args.resume = 'log/1027/SCOPSP_9class_wildceleba_percep-1.0x_clnew-1.5x_bg-0.3x_fd-0.1x_fo-0.1x_32x32_2/models/ep30.pkl'

    args = args_model = SLConfig.fromfile('log/1103/SCOPSP_9class_wildceleba_percep-1.5x_clnew-1.5x_bg-0.3x_arc+-1.0x_32x32/config-SCOPSP_WILDCELEBA_BG_Norm_128x128_TPS_PERCEPNEW_ARC_10lm_hm32_newtc.py')
    args_data = SLConfig.fromfile('configs/SCOPSP/TEST_wildceleba_SCOPS_DATASET.py')
    args.resume = 'log/1103/SCOPSP_9class_wildceleba_percep-1.5x_clnew-1.5x_bg-0.3x_arc+-1.0x_32x32/models/ep60.pkl'

    # config log    
    """LOGGER"""
    sys.stdout = Logger(osp.join(args.save_dir, 'test.log'))
    writer = SummaryWriter(osp.join(args.save_dir, 'writer'))
    os.makedirs(osp.join(args.save_dir, 'models'), exist_ok=True)
    os.makedirs(osp.join(args.save_dir, 'pics'), exist_ok=True)

    """DATA"""
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
    run_regression(model, dataloaderdict, writer, args, dtype=args.get('reg_list', 'face'))

def get_lm(model, dataloader, args, dtype='face'):
    # init
    _ = model.eval()
    batches = tqdm(dataloader, total=len(dataloader), position=0, leave=True, ascii=True)
    
    lm_gt_all = None
    lm_pred_all = None
    if dtype == 'CUB':
        bbox_all = None

    # gen json lm
    for idx, sample in enumerate(batches):
        # get data
        image = sample['image'].cuda()
        lm_gt = sample['landmarks'].numpy()

        ddata = {
            'img': image,
        }
        output = model(ddata, mode='get_hm_pred', get_loss=False)['output']
        hm_sm = output['hm_sm'] # B,N,H,W
        hm_sm = (hm_sm.max(1, keepdim=True)[0] == hm_sm).type(torch.float)
        lm_unsup = get_expected_points_from_map(hm_sm)


        
        # save the data
        if lm_gt_all is None:
            lm_gt_all = lm_gt
        else:
            lm_gt_all = np.concatenate((lm_gt_all, lm_gt), 0)
        
        lm_unsup = lm_unsup.detach().cpu().numpy().astype(lm_gt.dtype)
        if lm_pred_all is None:
            lm_pred_all = lm_unsup
        else:
            lm_pred_all = np.concatenate((lm_pred_all, lm_unsup), 0) 

        if dtype == 'CUB':
            figsize = sample['imgsize'] 
            bbox = sample['bbox'][:,2:]
            assert bbox.shape == figsize.shape
            bbox = bbox / figsize
            if bbox_all is None:
                bbox_all = bbox.numpy()
            else:
                bbox_all = np.concatenate((bbox_all, bbox), 0) 

    if dtype == 'CUB':
        return lm_pred_all, lm_gt_all, bbox_all
    else:
        return lm_pred_all, lm_gt_all, None

def reg_results(X_train, y_train, X_test, y_test, args, dtype='face', bbox=None):
    # print('-'*12, '\n', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # regression
    bias = args.get('bias', False)
    regr = linear_model.Ridge(alpha=0.0, fit_intercept=bias)
    _ = regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)

    landmarks_gt = y_test # B,N,2
    landmarks_regressed = y_predict.reshape(landmarks_gt.shape)

    # normalized error with respect to intra-occular distance
    if dtype in ['face', 'wildceleba', 'celeba', 'AFLW', 'wildAFLW']:
        eyes = landmarks_gt[:, :2, :]
        occular_distances = np.sqrt(
            np.sum((eyes[:, 0, :] - eyes[:, 1, :])**2, axis=-1))
        distances = np.sqrt(np.sum((landmarks_gt - landmarks_regressed)**2, axis=-1))
        mean_error = np.mean(distances / occular_distances[:, None])
    elif dtype == 'CUB':
        # print(landmarks_gt.shape, landmarks_regressed.shape, bbox.shape)
        landmarks_gt = landmarks_gt / bbox
        landmarks_regressed = landmarks_regressed / bbox
        distances = np.sqrt(np.sum((landmarks_gt - landmarks_regressed)**2, axis=-1))
        # distances = distances / 
        mean_error = np.mean(distances)
    else:
        raise ValueError('Unknown dtype %s' % dtype)

    # result
    # print('regression result:')
    # print('mean_error:', mean_error)

    return mean_error




def run_regression(model, dataloaderdict, writer, args, dtype='face'):
    print('=======> Regression', flush=True)
    print('run_regression 1: save data')
    # set the save path
    # with dataloaderdict['train'].dataset.raw_img_pred():
    X_train, y_train, bbox_train = get_lm(model, dataloaderdict['train'], args, dtype=dtype)
    # with dataloaderdict['test'].dataset.raw_img_pred():
    X_test, y_test, bbox_test = get_lm(model, dataloaderdict['test'], args, dtype=dtype)

    print('type\tshape')
    print(f'X_train.shape\t{X_train.shape}')
    print(f'y_train.shape\t{y_train.shape}')
    print(f'X_test.shape\t{X_test.shape}')
    print(f'y_test.shape\t{y_test.shape}')

    # np.save('tmp/regdata_1029.npy', {
    #     'X_train': X_train,
    #     'y_train': y_train,
    #     'X_test': X_test,
    #     'y_test': y_test
    # })

    # X_train = X_train.reshape(X_train.shape[0], -1)
    # y_train = y_train.reshape(y_train.shape[0], -1)
    # X_test = X_test.reshape(X_test.shape[0], -1)

    # print('run_regression 2: run reg')
    # # regression
    # bias = args.get('bias', False)
    # regr = linear_model.Ridge(alpha=0.0, fit_intercept=bias)
    # _ = regr.fit(X_train, y_train)
    # y_predict = regr.predict(X_test)

    # landmarks_gt = y_test # B,N,2
    # landmarks_regressed = y_predict.reshape(landmarks_gt.shape)

    # # normalized error with respect to intra-occular distance
    # eyes = landmarks_gt[:, :2, :]
    # occular_distances = np.sqrt(
    #     np.sum((eyes[:, 0, :] - eyes[:, 1, :])**2, axis=-1))
    # distances = np.sqrt(np.sum((landmarks_gt - landmarks_regressed)**2, axis=-1))
    # mean_error = np.mean(distances / occular_distances[:, None])

    # # result
    # print('regression result:')
    # print('mean_error:', mean_error)
    if dtype in ['face', 'wildceleba', 'celeba', 'AFLW', 'wildAFLW']:
        mean_error = reg_results(X_train, y_train, X_test, y_test, args, dtype=dtype, bbox=None)
    elif dtype == 'CUB':
        mean_error_con = AverageMeter()
        for i in range(y_train.shape[1]):
            y_train_i = y_train[:, i, :]
            y_test_i = y_test[:, i, :]
            train_idxs = np.where(y_train_i[:,0] > 0)
            test_idxs = np.where(y_test_i[:,0] > 0)
            if train_idxs[0].shape[0] == 0 or test_idxs[0].shape[0] == 0:
                continue
            X_train_used = X_train[train_idxs]
            y_train_used = y_train_i[train_idxs][:,np.newaxis,:]
            X_test_used = X_test[test_idxs]
            y_test_used = y_test_i[test_idxs][:,np.newaxis,:]
            if bbox_test is not None:
                bbox_test_used = bbox_test[test_idxs]
            else:
                bbox_test_used = None
            mean_error = reg_results(X_train_used, y_train_used, X_test_used, y_test_used, args, dtype=dtype, bbox=bbox_test_used)
            mean_error_con.update(mean_error)
        mean_error = mean_error_con.avg
    else:
        raise ValueError('Unknown dtype %s' % dtype)

    print('regression result:')
    print('mean_error:', mean_error)

    return mean_error

if __name__ == "__main__":
    main()