
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

def main():
    args = args_model = SLConfig.fromfile('2021log/0108/IMM_10class_wildceleba_saliency/config-MyIMM_WILDCELEBA_128x128_TPS_PERCEP_10lm_h32_mask.py')
    args.MODELNAME = 'MyIMM'
    args_data = SLConfig.fromfile('configs/SCOPSP/TEST_wildceleba_SCOPS_DATASET.py')
    args.resume = '2021log/0108/IMM_10class_wildceleba_saliency/models/test_best.pkl'

    # config log    
    """LOGGER"""
    sys.stdout = Logger(osp.join(args.save_dir, 'testwildcelebalast.log'))
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
    run_regression(0, model, dataloaderdict, writer, args)

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

        if args.MODELNAME == 'MyIMM' or args.MODELNAME == 'IMMSC' \
            or args.MODELNAME == 'MyIMMPP' or args.MODELNAME == 'MyIMMBN' :
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


def run_regression(epoch, model, dataloaderdict, writer, args):
    print('=======> Regression', flush=True)
    print('run_regression 1: save data')
    tmp = dataloaderdict['train'].dataset.pic_trans_num
    dataloaderdict['train'].dataset.pic_trans_num = 0
    # set the save path
    # with dataloaderdict['train'].dataset.raw_img_pred():
    X_train, y_train = get_lm(model, dataloaderdict['train'], args)
    # with dataloaderdict['test'].dataset.raw_img_pred():
    X_test, y_test = get_lm(model, dataloaderdict['test'], args)

    dataloaderdict['train'].dataset.pic_trans_num = tmp

    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    print('type\tshape')
    print(f'X_train.shape\t{X_train.shape}')
    print(f'y_train.shape\t{y_train.shape}')
    print(f'X_test.shape\t{X_test.shape}')
    print(f'y_test.shape\t{y_test.shape}')

    print('run_regression 2: run reg')
    # regression
    bias = args.get('bias', False)
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

    # save
    if writer is not None:
        writer.add_scalar('Regression/MSE', mean_error, epoch + 1)

    return mean_error


    
if __name__ == "__main__":
    main()

    
