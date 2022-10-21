
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(sys.path[0]))

from torch.utils.tensorboard.writer import SummaryWriter
from utils.utils import Logger
from utils.slconfig import SLConfig

import numpy as np
from sklearn import linear_model

import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.plot_landmarks import plot_landmarks
from network.modelwarper import ModelWarperV2
from datasets.get_loader import get_dataloader

# CUDA_VISIBLE_DEVICES 

# datatype = 'wildceleba'
datatype = 'AFLW'

def main():
    args_data = SLConfig.fromfile(f'configs/SCOPSP/TEST_{datatype}_SCOPS_DATASET.py')
    # 'configs/SCOPSP/TEST_wildceleba_SCOPS_DATASET.py'

    """DATA"""
    train_loader, test_loader, train_sampler = get_dataloader(args_data)
    print('len(train_set):', len(train_loader.dataset))
    print('len(test_set):', len(test_loader.dataset))

    dataloaderdict = {
        'train': train_loader,
        'test': test_loader,
    }
    save_data(0, dataloaderdict)

def get_lm(dataloader):
    # init
    batches = tqdm(dataloader, total=len(dataloader), position=0, leave=True, ascii=True)
    
    lm_gt_all = None

    # gen json lm
    for idx, sample in enumerate(batches):
        # get data
        # image = sample['image'].cuda()
        lm_gt = sample['landmarks'].numpy()
        
        # save the data
        if lm_gt_all is None:
            lm_gt_all = lm_gt
        else:
            lm_gt_all = np.concatenate((lm_gt_all, lm_gt), 0)

    return lm_gt_all


def save_data(epoch, dataloaderdict):
    print('=======> Regression', flush=True)
    print('run_regression 1: save data')
    tmp_train = dataloaderdict['train'].dataset.pic_trans_num
    tmp_test = dataloaderdict['test'].dataset.pic_trans_num
    dataloaderdict['train'].dataset.pic_trans_num = 0
    dataloaderdict['test'].dataset.pic_trans_num = 0

    print('len(train)', len(dataloaderdict['train'].dataset))
    print('len(test)', len(dataloaderdict['test'].dataset))

    y_train = get_lm(dataloaderdict['train'])
    y_test = get_lm(dataloaderdict['test'])

    print('type\tshape')
    print(f'y_train.shape\t{y_train.shape}')
    print(f'y_test.shape\t{y_test.shape}')

    os.makedirs(f'tmp/{datatype}', exist_ok=True)
    os.makedirs(f'tmp/{datatype}', exist_ok=True)

    np.save(f'tmp/{datatype}/training.npy', y_train)
    np.save(f'tmp/{datatype}/testing.npy', y_test)



    
if __name__ == "__main__":
    main()

    
