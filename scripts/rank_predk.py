import os, sys
import os.path as osp
sys.path.append(osp.dirname(sys.path[0]))

import numpy as np
from sklearn import linear_model

import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.plot_landmarks import plot_landmarks
from configs.myconfig import get_config
from datasets.get_loader import get_dataloader
from network.modelwarper import ModelWarperV2

import matplotlib.pyplot as plt
import matplotlib as mpl


def get_lm(model, dataloader, args):
    # init
    _ = model.eval()
    batches = tqdm(dataloader, total=len(dataloader), position=0, leave=True, ascii=True)
    
    lm_gt_all = None
    lm_pred_all = None

    # gen json lm
    for idx, sample in enumerate(batches):
        # get data
        image = sample['image'].cuda()
        lm_gt = sample['landmarks'].numpy()

        if args.MODELNAME == 'MyIMM' or args.MODELNAME == 'IMMSC' or args.MODELNAME == 'MyIMMPP':
            ddata = {
                'img': image,
            }
            output = model(ddata, mode='get_lm_pred', get_loss=False)['output']
            lm_unsup = output['lm_y']
            if 'pb_y' in output:
                pb_y = output['pb_y']
                lm_unsup = torch.cat((lm_unsup, pb_y.squeeze(-1)), -1)
        else:
            raise NotImplementedError("unsupported model name: %s" % args.MODELNAME)
        
        

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

    return lm_pred_all, lm_gt_all


def run_plot(epoch, model, dataloaderdict, args):
    print(' 1: save data')
    # set the save path
    with dataloaderdict['train'].dataset.raw_img_pred():
        X_train, y_train = get_lm(model, dataloaderdict['train'], args)

    thesavedir = 'tmp/rank_predkp'

    ds = dataloaderdict['train'].dataset
    for i in range(10):
        v = X_train[:,i,:]
        sortres = v[:,2].argsort()[::-1]
        length = len(sortres)
        for j, idx in enumerate(sortres):
            if j > 300 and j < length - 300:
                continue
            sample = ds[idx]
            img = sample['image'].permute(1,2,0).numpy()
            lm = v[idx][:2] * 128
            pred = v[idx][2]

            savedir = osp.join(thesavedir, "feat%d" % i) 
            os.makedirs(savedir, exist_ok=True)
            # cv2.imwrite("tmp/rank_plot/feat%d/raw%d-%d.jpg" % (i, j, idx), img[..., ::-1])
            # img = cv2.circle(img.astype(np.float32), (int(lm[i,0]), int(lm[i,1])), 1, (0,0,255), -1)
            # cv2.imwrite("tmp/rank_plot/feat%d/%d-%d.jpg" % (i, j, idx), img[..., ::-1])
            plt.imshow(img)
            plt.scatter(int(lm[0]), int(lm[1]), s=(5 * mpl.rcParams['lines.markersize']) ** 2)
            plt.axis('off')
            plt.title('score:%0.4f'% pred)
            plt.savefig(osp.join(savedir, '%d-%d-%0.4f.jpg' % (j, idx, pred)))
            plt.close()



def main(args):
    """MODEL"""
    model = ModelWarperV2(args)
    # for k,v in model.named_modules():
    #     print(k)
    # raise ValueError
    model = model.cuda()

    model = nn.DataParallel(model)

    """RESUME & PRETRAINED"""
    if args.get('resume', None) is not None:
        print("Loading checkpoint from '{}'".format(args.resume), flush=True)
        checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda())
        # load state dict to base model
        model.module.model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        raise ValueError('No resume model')

    rg_train_loader, rg_test_loader, rg_train_sampler = get_dataloader(args.REGRESSION_MAFL)
    dataloaderdict = {
        'train': rg_train_loader,
        'test': rg_test_loader
    }
    run_plot(0, model, dataloaderdict, args.REGRESSION_MAFL)


if __name__ == "__main__":
    args = get_config(copycfg=False)
    main(args)

        

    
