import os, sys
from os.path import join
import os.path as osp

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from datasets.basedataset import BaseDataSet

class WildAFLW(BaseDataSet, Dataset):
    def __init__(self, root_dir,
            select_path=None,
            anno_path=None,
            anno_norm=False,
            bbox_path=None,
            
            transform=None,
            pic_trans_num:int=0, pic_trans_type=None, pic_trans_cont=False, 
                pic_return_mask=False, soft_mask=True,
            skip_blank=True,
            select_larger_face=True,
            args=None,
            **kw):
        self.root_dir = root_dir
        self.select_path = select_path
        self.anno_path = anno_path
        self.anno_norm = anno_norm
        self.bbox_path = bbox_path
        self.transform = transform
        self.pic_trans_num = pic_trans_num
        self.pic_trans_type = pic_trans_type
        self.pic_trans_cont = pic_trans_cont
        self.pic_return_mask = pic_return_mask
        self.soft_mask = soft_mask
        self.skip_blank = skip_blank
        self.select_larger_face = select_larger_face

        self.datalist = self.pre_process()
        self.post_process()

    def pre_process(self):
        # open imglist
        if self.select_path is None:
            dirlist = os.listdir(self.root_dir)
            imglist = []
            for dirname in dirlist:
                imglist.extend([osp.join(dirname, i) for i in os.listdir(osp.join(self.root_dir, dirname))])
            imglist.sort()
        else:
            imglist = [line.strip() for line in open(self.select_path, 'r')]
        # print('len(imglist):', len(imglist))

        # result datalist
        datalist = []
        
        absvar = lambda x: (np.abs(x - x.mean(0)[np.newaxis,:])).sum()
        # open anno_path and select_path
        if self.anno_path is not None:
            annolist = [line.strip() for line in open(self.anno_path, 'r')]
            annolist = annolist[2:]
            annodict = {}
            for anno in annolist:
                coords = [x.strip() for x in anno.split(' ') if x.strip() != '']
                k = coords[0]
                v = np.array([float(x) for x in coords[1:]]).reshape(-1,2)
                if v.min() < 0:
                    continue
                if k not in annodict:
                    annodict[k] = [v]
                else:
                    del annodict[k]
                    continue
                    # if self.select_larger_face:
                    #     if absvar(v) > absvar(annodict[k][0]):
                    #         annodict[k] = [v]
                    # else:
                    #     annodict[k].append(v)
        else:
            annolist = None # list(range(len(imglist)))
            annodict = None


        if self.bbox_path is not None:
            bboxlist = [line.strip() for line in open(self.bbox_path, 'r')]
            bboxlist = bboxlist[2:]
            bboxdict = {}
            for bbox in bboxlist:
                coords = [x.strip() for x in bbox.split(' ') if x.strip() != '']
                k = coords[0]
                v = np.array([float(x) for x in coords[1:]]).reshape(-1,2)
                if k not in bboxdict:
                    bboxdict[k] = [v]
                else:
                    bboxdict[k].append(v)
        else:
            bboxlist = None # list(range(len(imglist)))
            bboxdict = None

        # iter 
        for idx, imgname in enumerate(imglist):
            # adding to the datalist
            item = {
                'imgpath': osp.join(self.root_dir, imgname),
                'imgname': imgname,
            }

            if self.anno_path is not None:
                annos = annodict.get(imgname, None)
                if annos is None:
                    continue
                for i in range(len(annos)):
                    itemi = item.copy()
                    itemi['landmarks'] = annos[i]
                    datalist.append(itemi)
            else:
                datalist.append(item)

            # ! TODO: process multi bbox

        # print(len(datalist))
        return datalist

    def __len__(self):
        return len(self.datalist)

    def get_metadata(self, idx):
        return self.datalist[idx]
        
