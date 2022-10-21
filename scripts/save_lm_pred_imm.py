import os, sys
import os.path as osp
import datetime
from tqdm import tqdm
import random
import json

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from torch.cuda.amp import GradScaler, autocast

from configs.myconfig import get_config
from datasets.get_loader import get_dataloader
from network.modelwarper import ModelWarper, ModelWarperV2
from regression.regression_sklearn import run_regression
from utils.utils import AverageMeter, FeiWu, Logger
from utils.plot_landmarks import plot_landmarks, unfold_heatmap

import matplotlib.pyplot as plt


def main(args):
    args.gpu_number = len(args.gpu_devices.split(','))
    args.world_size = args.gpu_number * args.nodes

    run(0, args)
    
def run(gpuid, args):
    """distribution init"""
    args.gpuid = gpuid

    """CONFIG"""
    torch.backends.cudnn.benchmark = True
    # only have output on gpu:0

    # some logs
    starttime = datetime.datetime.now()
    print("Start time: ", starttime, flush=True)
    print('command: python %s' % args.command, flush=True)
    print(args.pretty_text, flush=True)

    """DATA"""
    train_loader, test_loader, train_sampler = get_dataloader(args)
    print('len(train_set):', len(train_loader.dataset))
    print('len(test_set):', len(test_loader.dataset))

    """MODEL"""
    model = ModelWarperV2(args)
    # for k,v in model.named_modules():
    #     print(k)
    # raise ValueError
    model = model.cuda(gpuid)

    if args.get('distributed', False):
        model = nn.DataParallel(model, device_ids=[gpuid])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpuid], find_unused_parameters=True)
    else:
        model = nn.DataParallel(model)
        

    """RESUME & PRETRAINED"""
    if args.get('resume', None) is not None:
        print("Loading checkpoint from '{}'".format(args.resume), flush=True)
        checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(gpuid))
        # load state dict to base model
        model.module.model.load_state_dict(checkpoint['state_dict'], strict=False)

    """RUN"""

    save_points(model, test_loader, gpuid, args)

    endtime = datetime.datetime.now()
    print("Total time used: {}".format( endtime - starttime ), flush=True)

def save_points(model, test_loader, gpuid, args):
    mixed_precision = args.get('mixed_precision', False)
    _ = model.eval()

    print(f'=======> Save lm to {args.savepath}', flush=True)
    with torch.no_grad():
        if args.gpuid == 0:
            batches = tqdm(test_loader, total=len(test_loader), position=0, leave=True, ascii=True)
        else:
            batches = test_loader
        for idx, sample in enumerate(batches):

            # get data
            image = sample['image'].cuda()
            imgname = sample['imgname']

            ddata = {
                'img': image,
            }

            # throw them into the model
            with autocast(enabled=mixed_precision):
                lm_pred = model(ddata, mode='get_lm_pred', get_loss=False)['output']['lm_y']

            if args.testmode:
                if idx > 5:
                    break

            lm_pred = lm_pred.detach().cpu().numpy()

            for lm_i, name_i in zip(lm_pred, imgname):
                name_i = osp.basename(name_i)
                with open(args.savepath, 'a') as f:
                    f.write(json.dumps({
                            'imagename': name_i,
                            'landmarks': lm_i.tolist()
                        }) + '\n')


if __name__ == "__main__":
    args = get_config()
    main(args)