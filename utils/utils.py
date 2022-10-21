from __future__ import absolute_import
import os
from random import random
import sys
import errno
import shutil
import json
from typing import List
import torch
import os.path as osp
import torch.nn as nn
from sklearn import metrics
import numpy as np


def gifdict(d, kname, default='Error'):
    if kname is None:
        return None
    kname_list = [i.strip() for i in kname.split('.')]
    res = d
    for i in kname_list:
        if isinstance(res, dict):
            res = res.get(i, default)
        elif isinstance(res, list):
            res = res[int(i)]
        else:
            raise ValueError("Unsupported type: {}".format(type(res)))
        if res == 'Error':
            raise ValueError('%s unfound in keys:' % kname, ' '.join([str(ii) for ii in d.keys()]))
    return res


def squeeze_recur(x, dim):
    if isinstance(x, torch.Tensor):
        return x.squeeze(dim)
    if isinstance(x, List) or isinstance(x, tuple):
        return [squeeze_recur(i, dim) for i in x]
    if isinstance(x, dict):
        return {
            k:squeeze_recur(v, dim) 
            for k,v in x.items()
        }


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    """
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    @property
    def _avg(self):
        if self.cnt == 0:
            return 1
        return self.avg


def presentParameters(args_dict):
    """
        Print the parameters setting line by line
        Arg:    args_dict   - The dict object which is transferred from argparse Namespace object
    """
    print("========== Parameters ==========")
    for key in sorted(args_dict.keys()):
        print("{:>15} : {}".format(key, args_dict[key]))
    print("===============================")


def stat_thop(model, input):
    # from thop import profile
    # flops, params = profile(model, inputs=(input, ))
    # print('flops:%s\nparams:%s' % (flops, params))

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("params: %0.6fM" % (num_params / 1e6))


class FeiWu():
    def __init__(self, *attr, **kw):
        pass

    def passfunc(self, *attr, **kw):
        pass
    def __getattr__(self, x):
        return self.passfunc

    def __enter__(self, *attr, **kw):
        pass

    def __exit__(self, *attr, **kw):
        pass
        

# points perturb
def get_rand_delta(x):
    """[summary]

    Args:
        x (np.array): N
    """
    p = (random() * 2 - 1) * 0.1
    # minx, maxx = x.min(), x.max()
    # if p + maxx - minx > 0.85:
    #     p = 0.85 - maxx + minx
    # if p < 0.15:
    #     p = 0.15
    # return p - minx
    return p

def points_pertub(x):
    """[summary]

    Args:
        x (np.array): N,2
    """
    deltax = get_rand_delta(x[:,0])
    deltay = get_rand_delta(x[:,1])
    return x + (deltax, deltay)

def points_symmic_pertub(img1, img2):
    """[summary]

    Args:
        x (np.array): N,2
    """
    deltax = get_rand_delta(img1[:,0])
    deltay = get_rand_delta(img1[:,1])


    return img1 + (deltax, deltay), img2 + (-deltax, deltay)

# map from lm to color

def get_lm_map():
    lmmap = {}
    lmmap.update({i:6 for i in range(6)})
    lmmap.update({i:5 for i in range(6, 11)})
    lmmap.update({i:8 for i in range(11, 17)})
    lmmap.update({i:3 for i in range(17, 29)})
    lmmap.update({i:2 for i in range(29, 37)})
    lmmap.update({i:3 for i in range(37, 49)})
    lmmap.update({i:2 for i in range(49, 68)})
    return lmmap


## colorize
class Colorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_color_index_map(img, index_list=None):
    """[summary]

    Args:
        img (C,H,W): [description]
        index ([type], optional): [description]. Defaults to None.
    """
    if index_list == None:
        index_list = range(img.shape[0])
    idxmap = np.argmax(img, 0)
    res = np.concatenate([(idxmap == i).astype(np.float32)[np.newaxis, :, :] for  i in index_list], 0)
    return res
    


def slprint(x, name='x'):
    if isinstance(x, (torch.Tensor, np.ndarray)):
        print(f'{name}.shape:', x.shape)
    elif isinstance(x, (tuple, list)):
        print('type x:', type(x))
        for i in range(min(10, len(x))):
            slprint(x[i], f'{name}[{i}]')
    elif isinstance(x, dict):
        for k,v in x.items():
            slprint(v, f'{name}[{k}]')
    else:
        print(f'{name}.type:', type(x))