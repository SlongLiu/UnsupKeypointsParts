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
parser.add_argument('--last_layer', '-l', type=str, default='relu5_2')
parser.add_argument('--n_instance', type=int, default=10000, help='number of instance')
parser.add_argument('--feat_path', type=str, default='features2.pkl', help='number of instance')
parser.add_argument('--lm_path', type=str, default='lm2.pkl', help='number of instance')
args = parser.parse_args()



os.environ['CUDA_VISIBLE_DEVICES'] = '7'
last_layer = args.last_layer
use_cuda = True
# resume = 'log/1007/IMMSC_centerloss-1x_celeba_10lm_0.02hm_RST_h32_r100_lr1e-4_relu5_2_resume/models/ep49.pkl'
# cfgpath = 'configs/IMMSC/MyIMM_CELEBA_v3_128x128_TPS_PERCEP_SC_10lm_h32_newtc.py'

resum = 'log/1007/IMMSC_centerLineloss-1x_celeba_10lm_0.02hm_RST_h32_r100_lr1e-4_pool4_resume/models/ep73.pkl'
cfgpath = 'configs/IMMSC/MyIMM_CELEBA_v3_128x128_TPS_PERCEP_SCLine_10lm_h32_newtc.py'


# resume = 'log/0912/IMM_celeba_v8_10lm_0.02hm_tps2_percept_h32_r100_1x_lr1e-4_point_softmask_2/models/ep60.pkl'
# cfgpath = 'configs/IMM/MyIMM_CELEBA_v3_128x128_TPS_PERCEP_10lm_h32_newtc.py'

# resume = 'log/1002/IMMSC+triplet_1.0x_celeba_10lm_0.02hm_RST_h32_r100_lr1e-4_pool4_resume/models/ep28.pkl'
# cfgpath = 'configs/IMM/MyIMM_CELEBA_v3_128x128_TPS_PERCEP_SCtri_10lm_h32_newtc.py'

# resume = 'log/1002/IMMSC_newpercep-1x_tripletcm-1x_10lm_0.02hm_RST_h32_r100_lr1e-4/models/test_best.pkl'
# resume = 'log/1003/IMMSC_newpercep-1x_tripletcm-1x_10lm_0.02hm_RST_h32_r100_lr1e-4_cont/models/ep35.pkl'
# cfgpath = 'configs/IMM/MyIMM_CELEBA_v3_128x128_TPS_PERCEPNEW_SCtri_10lm_h32_newtc.py'
feat_type = 'after'
thesavedir = 'tmp/new_rank_plot/l2_clline_model_after'
featpath = osp.join(thesavedir, args.feat_path)
lmpath = osp.join(thesavedir, args.lm_path)



# dataset
myargs = SLConfig.fromfile(cfgpath) 
ds = get_dataset_by_config(myargs)[0]
ds.pic_trans_num = 0


def norm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(img.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(img.device)
    img = (img.permute(0, 2, 3, 1) - mean) / std
    return img.permute(0, 3, 1, 2)


def iter_mean(feat, stat=True, n=5, therhold=0.95):
    """[summary]

    Args:
        feat ([type]): B,N,512
        n (int, optional): [description]. Defaults to 3.
        ratio (float, optional): [description]. Defaults to 0.7.
    
    Return:
        mean_feat: 
    """
    feat_raw = feat
    mean_feat_list = []
    cos_sim_list = []
    for i in range(feat.shape[1]):
        print('\nFeature %d\n' % i)
        feat_raw_i = feat_raw[:, i, :]
        idxs = torch.arange(feat_raw_i.shape[0])    
        for j in range(n):
            feat = feat_raw[idxs, i] # B,512
            mean_feat = feat.mean(0, keepdim=True) # 1, 512

            # print(feat.shape, feat_raw_i.shape, mean_feat.shape)
            cos_sim = (feat_raw_i * mean_feat).sum(-1) / torch.norm(feat_raw_i, dim=-1) / torch.norm(mean_feat, dim=-1) # B
            # print('cos_sim.shape:', cos_sim.shape)
            
            # stat
            print('Step %d' % j, 'total: %d' % feat.shape[0])
            if stat:
                # for j in range(cos_sim.shape[-1]):
                hist, _ = np.histogram(cos_sim, bins = np.linspace(0.5,1,21))
                print("%d:\t" % j + "\t".join([str(ii) for ii in hist.tolist()]))

            idxs = torch.where(cos_sim > 0.8)[0]
        mean_feat_list.append(mean_feat.unsqueeze(1))
        cos_sim_list.append(cos_sim.unsqueeze(1))
    mean_feat_res = torch.cat(mean_feat_list, dim=1)
    cos_sim_res = torch.cat(cos_sim_list, dim=1)

    return mean_feat_res, cos_sim_res

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

        ddata = {
            'img': img
        }

        output = model(ddata, mode='get_lm_pred', get_loss=False)
        lm = output['output']['lm_y']
        hmr = output['output']['hmr_y']
        hm = output['output']['hm_y']
        # print('lm.shape:', lm.shape)

        fe = fenet(norm(img))
        fe = torch.cat(fe, dim=1)

        # hm = get_gaussian_map_from_points(lm, 16, 16, 0.02, lm.device, mode='point')
        B,C,H,W = hm.shape
        # hm = nn.functional.softmax(hm.reshape(B, C, -1), dim=1).reshape(B,C,H,W)
        if feat_type == 'after':
            lf = (hm.unsqueeze(2) * fe.unsqueeze(1)).sum((-1, -2))
        elif feat_type == 'before':
            lf = (hmr.unsqueeze(2) * fe.unsqueeze(1)).sum((-1, -2))
        else:
            raise ValueError

        # save
        lfc = lf.detach().cpu().clone()
        lfc.requires_grad = False
        if warehouse is None:
            warehouse = lfc
        else:
            warehouse = torch.cat((warehouse, lfc), axis=0) # N, 10, 512

        lmc = lm.detach().cpu().clone()
        lmc.requires_grad = False
        if lmhouse is None:
            lmhouse = lmc
        else:
            lmhouse = torch.cat((lmhouse, lmc), axis=0) # N,10,2

    os.makedirs(osp.dirname(featpath), exist_ok=True)
    os.makedirs(osp.dirname(lmpath), exist_ok=True)
    torch.save(warehouse, featpath)
    torch.save(lmhouse, lmpath)
    return warehouse, lmhouse


def find_best(mean_feat, lms, cos_sim, n=300):
    """[summary]

    Args:
        mean_feat ([type]): [description]
        lms ([type]): [description]
        cos_sim ([type]): B,10
        n (int, optional): [description]. Defaults to 100.
    """
    for i in range(cos_sim.shape[1]):
        sim_i = cos_sim[:,i]
        scores, idxs = torch.sort(sim_i)

        for j in range(n):
            idx = idxs[j]
            score = scores[j].item()
            sample = ds[idx]
            img = sample['image'].permute(1,2,0).numpy()
            lm = lms[idx] * 128
            # img = np.array(img*255).astype(int)
            savedir = osp.join(thesavedir, "feat%d" % i) 
            os.makedirs(savedir, exist_ok=True)
            # cv2.imwrite("tmp/rank_plot/feat%d/raw%d-%d.jpg" % (i, j, idx), img[..., ::-1])
            # img = cv2.circle(img.astype(np.float32), (int(lm[i,0]), int(lm[i,1])), 1, (0,0,255), -1)
            # cv2.imwrite("tmp/rank_plot/feat%d/%d-%d.jpg" % (i, j, idx), img[..., ::-1])
            plt.imshow(img)
            plt.scatter(int(lm[i,0]), int(lm[i,1]), s=(5 * mpl.rcParams['lines.markersize']) ** 2)
            plt.axis('off')
            plt.title('score:%0.4f'% score)
            plt.savefig(osp.join(savedir, '%d-%d-%0.4f.jpg' % (j, idx, score)))
            plt.close()
        
        for j in range(cos_sim.shape[0]-n, cos_sim.shape[0]):
            idx = idxs[j]
            score = scores[j].item()
            sample = ds[idx]
            img = sample['image'].permute(1,2,0).numpy()
            lm = lms[idx] * 128
            # img = np.array(img*255).astype(int)
            # img = cv2.circle(img, (int(lm[i,0]), int(lm[i,1])), 1, (0,0,255), -1)
            # cv2.imwrite("tmp/rank_plot/feat%d/%d-%d.jpg" % (i, j, idx), img[..., ::-1])
            plt.imshow(img)
            plt.scatter(int(lm[i,0]), int(lm[i,1]), s=(5 * mpl.rcParams['lines.markersize']) ** 2)
            plt.axis('off')
            plt.title('score:%0.4f'% score)
            # plt.savefig("tmp/rank_plot2/feat%d/%d-%d-%0.4f.jpg" % (i, j, idx, score))
            plt.savefig(osp.join(thesavedir, "feat%d" % i, '%d-%d-%0.4f.jpg' % (j, idx, score)))
            plt.close()


def main():
    if not os.path.exists(featpath):
        warehouse, lmhouse = get_feat()
    else:
        warehouse = torch.load(featpath)
        lmhouse = torch.load(lmpath)
    mean_feat, cos_sim = iter_mean(warehouse, n=1)
    find_best(mean_feat, lmhouse, cos_sim, n=300)


    # plot utils
    warehouse = torch.load(featpath)
    warehouse = warehouse[:1000]
    # print(warehouse.shape)
    mean_feat, cos_sim = iter_mean(warehouse)

    feats = torch.cat((warehouse, mean_feat), 0).numpy()
    cos_sims = torch.cat((cos_sim, torch.ones(1, 10)), 0).reshape(-1, 1).numpy()

    # plot
    c_1 = np.tile(np.arange(feats.shape[1]), feats.shape[0])
    cm = color_map(10)/255
    # print(cm[c_1].shape, cos_sims.shape)
    cm_1 = np.concatenate((cm[c_1], cos_sims), axis=-1)
    X = feats.reshape(-1, feats.shape[-1])
    print('X.shape:', X.shape, 'cm_1.shape:', cm_1.shape, 'cos_sims.shape:', cos_sims.shape)

    # tsne
    tsne = TSNE(n_components=2, init='pca')
    Y = tsne.fit_transform(X)
    
    plt.figure(figsize=(8,8), dpi=100)
    
    ax1 = plt.subplot(111)
    ax1.scatter(Y[:, 0], Y[:, 1], c=cm_1)
    ax1.scatter(Y[-10:, 0], Y[-10:, 1], s=100, c='white', marker='X')

    savepath = osp.join(thesavedir, 'newtsne-%s.jpg' % last_layer)
    os.makedirs(osp.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)


if __name__ == "__main__":
    main()