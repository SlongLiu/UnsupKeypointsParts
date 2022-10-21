import os
import os.path as osp

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


def run_regression(epoch, model, dataloaderdict, args):
    print('=======> Regression', flush=True)
    print('run_regression 1: save data')
    # set the save path
    with dataloaderdict['train'].dataset.raw_img_pred():
        X_train, y_train = get_lm(model, dataloaderdict['train'], args)
    with dataloaderdict['test'].dataset.raw_img_pred():
        X_test, y_test = get_lm(model, dataloaderdict['test'], args)

    
    np.save('tmp/xy_value.npy', {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    })
    
    mean_error = reg(X_train, y_train, X_test, y_test, bias=False)

    return mean_error

def reg(X_train, y_train, X_test, y_test, bias=False):
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    print('type\tshape')
    print(f'X_train.shape\t{X_train.shape}')
    print(f'y_train.shape\t{y_train.shape}')
    print(f'X_test.shape\t{X_test.shape}')
    print(f'y_test.shape\t{y_test.shape}')

    # print('run_regression 2: run reg')
    # regression
    # bias = args.get('bias', False)
    regr = linear_model.Ridge(alpha=0.0, fit_intercept=bias)
    _ = regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)

    landmarks_gt = y_test # B,N,2
    landmarks_regressed = y_predict.reshape(landmarks_gt.shape)

    # normalized error with respect to intra-occular distance
    eyes = landmarks_gt[:, :2, :]
    occular_distances = np.sqrt(
        np.sum((eyes[:, 0, :] - eyes[:, 1, :])**2, axis=-1))
    distances = np.sqrt(np.sum((landmarks_gt - landmarks_regressed)**2, axis=-1))
    mean_error = np.mean(distances / occular_distances[:, None])

    # result
    print('regression result:')
    print('mean_error:', mean_error)
    return mean_error


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
    run_regression(0, model, dataloaderdict, args.REGRESSION_MAFL)


if __name__ == "__main__":
    args = get_config(copycfg=False)
    main(args)

        

    
