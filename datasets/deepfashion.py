import os, sys
import os.path as osp
# import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from datasets.basedataset import BaseDataSet


class DeepFashion(BaseDataSet, Dataset):
    def __init__(self,
        root_dir,
        split,
        transform,
        dup = None,

        pic_trans_num:int=0, pic_trans_type=None, pic_trans_cont=False, 
            pic_return_mask=False, soft_mask=True,
        args=None,
        **kw,
    ):
        self.root_dir = root_dir
        self.split = split
        assert split in ['train', 'test', 'all']
        self.dup = dup

        self.transform = transform
        self.pic_trans_num = pic_trans_num
        self.pic_trans_type = pic_trans_type
        self.pic_trans_cont = pic_trans_cont
        self.pic_return_mask = pic_return_mask
        self.soft_mask = soft_mask

        self.datalist = self.pre_process()
        self.length = len(self.datalist)
        self.post_process()

    def pre_process(self):
        imglistpath = osp.join(self.root_dir, f'data_{self.split}.csv')
        if self.split == 'all':
            imglistpath = osp.join(self.root_dir, f'data.csv')
        imglist = [x.split(',')[-1].strip() for x in open(imglistpath, 'r')][1:]
        
        datalist = []

        for idx, imgname in enumerate(imglist):
            imgpath = osp.join(self.root_dir, imgname)
            item = {
                'imgname': imgname,
                'imgpath': imgpath
            }
            datalist.append(item)
        
        return datalist

    def __len__(self) -> int:
        if self.dup is None:
            return self.length
        else:
            return self.length * self.dup

    def get_metadata(self, idx):
        return self.datalist[idx % self.length]

                
                
                
