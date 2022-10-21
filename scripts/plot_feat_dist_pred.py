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
from network.modelwarper import ModelWarperV2

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--last_layer', '-l', type=str, default='pool4')
args = parser.parse_args()



os.environ['CUDA_VISIBLE_DEVICES'] = '2'
last_layer = args.last_layer
use_cuda = True
# resume = 'log/1002/IMMSC+triplet_0.1x_celeba_10lm_0.02hm_RST_h32_r100_lr1e-4_pool4/models/ep29.pkl'
# cfgpath = 'configs/IMM/MyIMM_CELEBA_v3_128x128_TPS_PERCEP_10lm_h32_newtc.py'
resume = 'log/0912/IMM_celeba_v8_10lm_0.02hm_tps2_percept_h32_r100_1x_lr1e-4_point_softmask_2/models/ep60.pkl'
cfgpath = 'configs/IMM/MyIMM_CELEBA_v3_128x128_TPS_PERCEP_10lm_h32_newtc.py'

def norm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(img.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(img.device)
    img = (img.permute(0, 2, 3, 1) - mean) / std
    return img.permute(0, 3, 1, 2)


def main():
    # dataset
    args = SLConfig.fromfile(cfgpath)
    ds = get_dataset_by_config(args)[0]
    ds.pic_trans_num = 0

    # model
    model = ModelWarperV2(args)
    model = model.cuda()
    print("Loading checkpoint from '{}'".format(resume), flush=True)
    checkpoint = torch.load(resume, map_location = lambda storage, loc: storage.cuda())
    # load state dict to base model
    model.model.load_state_dict(checkpoint['state_dict'], strict=False)
    _ = model.eval()

    fenet = FeatureExtraction(use_cuda=use_cuda, last_layer=last_layer)

    # iter
    warehouse = None

    for i in tqdm(range(100)):
        sample = ds[i]
        img = sample['image'].unsqueeze(0).cuda()

        ddata = {
            'img': img
        }

        output = model(ddata, mode='get_lm_pred', get_loss=False)
        lm = output['output']['lm_y']
        hmr = output['output']['hmr_y']
        # print('lm.shape:', lm.shape)

        fe = fenet(norm(img))
        fe = torch.cat(fe, dim=1)

        # hm = get_gaussian_map_from_points(lm, 16, 16, 0.02, lm.device, mode='point')
        lf = (hmr.unsqueeze(2) * fe.unsqueeze(1)).sum((-1, -2))

        # save
        lf = lf.detach().cpu().numpy()
        if warehouse is None:
            warehouse = lf
        else:
            warehouse = np.concatenate((warehouse, lf), axis=0)

    # plot
    c_1 = np.tile(np.arange(warehouse.shape[1]), warehouse.shape[0])
    c_2 = np.array(list(range(warehouse.shape[0]))).repeat(warehouse.shape[1])
    X = warehouse.reshape(-1, warehouse.shape[-1])
    print('X.shape:', X.shape, 'c_1.shape:', c_1.shape, 'c_2.shape:', c_2.shape)

    # tsne
    tsne = TSNE(n_components=2)
    Y = tsne.fit_transform(X)
    
    plt.figure(figsize=(8,12), dpi=100)
    
    ax1 = plt.subplot(211)
    ax1.scatter(Y[:, 0], Y[:, 1], c=c_1, cmap=plt.cm.Spectral)

    ax2 = plt.subplot(212)
    ax2.scatter(Y[:, 0], Y[:, 1], c=c_2, cmap=plt.cm.Spectral)

    os.makedirs('tmp/model3', exist_ok=True)
    savepath = "tmp/model3/tsne-%s.jpg" % last_layer
    plt.savefig(savepath)
    print('saved in %s' % savepath)


if __name__ == "__main__":
    main()