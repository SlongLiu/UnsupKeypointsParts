import os, sys
import os.path as osp

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from datasets.basedataset import BaseDataSet

class WildCelebA(BaseDataSet, Dataset):
    def __init__(self, root_dir,
            select_path=None,
            anno_path=None,
            anno_norm=False,
            bbox_path=None,
            transform=None,
            pic_trans_num:int=0, pic_trans_type=None, pic_trans_cont=False, 
                pic_return_mask=False, soft_mask=True,
            saliency_dir = None,
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
        self.saliency_dir = saliency_dir

        self.datalist = self.pre_process()
        self.post_process()

    def pre_process(self):
        # open imglist
        imglist = os.listdir(self.root_dir)
        imglist.sort()

        # result datalist
        datalist = []

        # open anno_path and select_path
        if self.anno_path is not None:
            annolist = [line.strip() for line in open(self.anno_path, 'r')]
            annolist = annolist[2:]
        else:
            annolist = None # list(range(len(imglist)))

        if self.bbox_path is not None:
            bboxlist = [line.strip() for line in open(self.bbox_path, 'r')]
            bboxlist = bboxlist[2:]
        else:
            bboxlist = None # list(range(len(imglist)))

        if self.select_path is not None:
            selectset = set([line.strip() for line in open(self.select_path, 'r')])
        else:
            selectset = None

        # print('len(selectset):', len(selectset))
        # print('len(annolist):', len(annolist))

        # iter
        for idx, (imgname, anno) in enumerate(zip(imglist, annolist)):
            # skip the imgs not in the selected list
            if selectset is not None and imgname not in selectset:
                continue
            
            # adding to the datalist
            item = {
                'imgpath': osp.join(self.root_dir, imgname),
                'imgname': imgname,
            }
            
            if self.anno_path is not None:
                # parser the anno
                anno = annolist[idx]
                coords = [x.strip() for x in anno.split(' ') if x.strip() != ''][1:]
                coords = np.array([float(x) for x in coords]).reshape(-1,2)
                item.update({'landmarks': coords})

            if self.bbox_path is not None:
                # parser the anno
                bbox = bboxlist[idx]
                bbox = [x.strip() for x in bbox.split(' ') if x.strip() != ''][1:]
                bbox = np.array([float(x) for x in bbox]).reshape(-1,2)
                item.update({'bbox': bbox})

            if self.saliency_dir is not None:
                item['saliency_path'] = osp.join(self.saliency_dir, imgname)

            datalist.append(item)

        # print('len(datalist):', len(datalist))
        return datalist

    def __len__(self):
        return len(self.datalist)

    def get_metadata(self, idx):
        return self.datalist[idx]
        
