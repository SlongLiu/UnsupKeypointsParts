from genericpath import exists
import os, sys
import os.path as osp

from matplotlib.pyplot import axis
sys.path.append(osp.dirname(sys.path[0]))

import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import cv2

import torchvision
from torchvision import models

from network.feature_extraction import FeatureExtraction
from utils.slconfig import SLConfig
from datasets.get_loader import get_dataset_by_config
from utils.plot_landmarks import plot_landmarks
from utils.soft_points import get_gaussian_map_from_points
from utils.utils import color_map
from network.modelwarper import ModelWarperV2

import matplotlib.pyplot as plt
import matplotlib as mpl

parser = argparse.ArgumentParser()
parser.add_argument('--last_layer', '-l', type=str, default='pool4')
parser.add_argument('--n_instance', type=int, default=-1, help='number of instance')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
last_layer = args.last_layer
use_cuda = True
resume = 'log/0912/IMM_celeba_v8_10lm_0.02hm_tps2_percept_h32_r100_1x_lr1e-4_point_softmask_2/models/ep60.pkl'
cfgpath = 'configs/IMM/MyIMM_CELEBA_v3_128x128_TPS_PERCEP_10lm_h32_newtc.py'

# resume = 'log/1002/IMMSC+triplet_1.0x_celeba_10lm_0.02hm_RST_h32_r100_lr1e-4_pool4_resume/models/ep28.pkl'
# cfgpath = 'configs/IMM/MyIMM_CELEBA_v3_128x128_TPS_PERCEP_SCtri_10lm_h32_newtc.py'

# resume = 'log/1002/IMMSC_newpercep-1x_tripletcm-1x_10lm_0.02hm_RST_h32_r100_lr1e-4/models/test_best.pkl'
# cfgpath = 'configs/IMM/MyIMM_CELEBA_v3_128x128_TPS_PERCEPNEW_SCtri_10lm_h32_newtc.py'
thesavedir = '/data/shilong/data/imm/celeba/predkp'
os.makedirs(thesavedir, exist_ok=True)

# dataset
myargs = SLConfig.fromfile(cfgpath) 
ds = get_dataset_by_config(myargs)[1]
ds.pic_trans_num = 0


def norm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(img.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(img.device)
    img = (img.permute(0, 2, 3, 1) - mean) / std
    return img.permute(0, 3, 1, 2)

def get_feat():
    # model
    model = ModelWarperV2(myargs)
    model = model.cuda()
    print("Loading checkpoint from '{}'".format(resume), flush=True)
    checkpoint = torch.load(resume, map_location = lambda storage, loc: storage.cuda())
    # load state dict to base model
    model.model.load_state_dict(checkpoint['state_dict'], strict=False)
    fenet = FeatureExtraction(use_cuda=use_cuda, last_layer=last_layer)

    # iter
    warehouse = None
    lmhouse = None
    total_num = args.n_instance
    if total_num == -1:
        total_num = len(ds)
    for i in tqdm(range(total_num)):
        sample = ds[i]
        img = sample['image'].unsqueeze(0).cuda()
        imgname = sample['imgname']

        ddata = {
            'img': img
        }

        output = model(ddata, mode='get_lm_pred', get_loss=False)
        lm = output['output']['lm_y']
        hmr = output['output']['hmr_y']
        hm = output['output']['hm_y']
        # print('lm.shape:', lm.shape)

        # save
        lm = lm[0].detach().cpu().numpy()
        savepath = osp.join(thesavedir, imgname+'.npy')
        np.save(savepath, lm)

        

    return warehouse, lmhouse



def main():
    get_feat()


if __name__ == "__main__":
    main()