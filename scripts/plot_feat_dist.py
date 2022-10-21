import os, sys
import os.path as osp
sys.path.append(osp.dirname(sys.path[0]))

import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE

import torchvision
from torchvision import models

from network.feature_extraction import FeatureExtraction
from utils.slconfig import SLConfig
from datasets.get_loader import get_dataset_by_config
from utils.plot_landmarks import plot_landmarks
from utils.soft_points import get_gaussian_map_from_points

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--last_layer', '-l', type=str, default='relu5_3')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
last_layer = args.last_layer
use_cuda=True

def norm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(img.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(img.device)
    img = (img.permute(0, 2, 3, 1) - mean) / std
    return img.permute(0, 3, 1, 2)


def main():
    # dataset
    args = SLConfig.fromfile('configs/IMM/MyIMM_CELEBA_v3_128x128_TPS_PERCEP_10lm_h32_newtc.py')
    ds = get_dataset_by_config(args)[0]
    ds.pic_trans_num = 0

    # model
    fenet = FeatureExtraction(use_cuda=use_cuda, last_layer=last_layer)

    warehouse = None

    for i in tqdm(range(100)):
        sample = ds[i]
        img = sample['image'].unsqueeze(0).cuda()
        lm = torch.Tensor(sample['landmarks']).unsqueeze(0).cuda()

        fe = fenet(norm(img))
        fe = torch.cat(fe, dim=1)

        hm = get_gaussian_map_from_points(lm, 16, 16, 0.02, lm.device, mode='point')
        lf = (hm.unsqueeze(2) * fe.unsqueeze(1)).sum((-1, -2))
        # norm
        # lf = lf / lf.

        # save
        lf = lf.detach().cpu().numpy()
        if warehouse is None:
            warehouse = lf
        else:
            warehouse = np.concatenate((warehouse, lf), axis=0)

    # np.save('warehouse.npy', warehouse)

    # plot
    c_1 = np.tile(np.array([1,2,3,4,5]), warehouse.shape[0])
    c_2 = np.array(list(range(warehouse.shape[0]))).repeat(warehouse.shape[1])
    X = warehouse.reshape(-1, warehouse.shape[-1])

    # tsne
    tsne = TSNE(n_components=2)
    Y = tsne.fit_transform(X)
    
    plt.figure(figsize=(8,12), dpi=100)
    
    ax1 = plt.subplot(211)
    ax1.scatter(Y[:, 0], Y[:, 1], c=c_1, cmap=plt.cm.Spectral)

    ax2 = plt.subplot(212)
    ax2.scatter(Y[:, 0], Y[:, 1], c=c_2, cmap=plt.cm.Spectral)

    os.makedirs('tmp/tsne', exist_ok=True)
    plt.savefig("tmp/tsne/tsne-%s.jpg" % last_layer)


if __name__ == "__main__":
    main()