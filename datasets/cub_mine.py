import os, sys
from os.path import join
import os.path as osp

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from datasets.basedataset import BaseDataSet

class CUB2(BaseDataSet, Dataset):
    def __init__(self,
        data_dir = '',
        select_class = None,
        split = 'all',
        use_crop = False,
        dup = None,

        transform=None,
        pic_trans_num:int=0, pic_trans_type=None, pic_trans_cont=False, 
                pic_return_mask=False, soft_mask=True,
        args=None,
        **kw):
        self.data_dir = data_dir
        self.select_class = select_class
        if isinstance(self.select_class, int):
            self.select_class = [self.select_class]
        self.split = split
        self.dup = dup
        self.use_crop = use_crop

        self.transform = transform
        self.pic_trans_num = pic_trans_num
        self.pic_trans_type = pic_trans_type
        self.pic_trans_cont = pic_trans_cont
        self.pic_return_mask = pic_return_mask
        self.soft_mask = soft_mask

        self.img_dir = osp.join(self.data_dir, 'images')
        self.img_list = osp.join(self.data_dir, 'images.txt')
        self.img_split = osp.join(self.data_dir, 'train_test_split.txt')
        self.anno = osp.join(self.data_dir, 'parts', 'part_locs.txt')
        self.bbox_file = osp.join(self.data_dir, 'bounding_boxes.txt')

        self.datalist = self.preprocess()
        self.length = len(self.datalist)
        self.post_process()

    def preprocess(self):
        imglist = [line.strip() for line in open(self.img_list)]
        bbox_list = [line.strip() for line in open(self.bbox_file)]
        train_test_split = [int(line.strip()[-1]) for line in open(self.img_split)]
        annolist = [[i.strip() for i in line.strip().split()] for line in open(self.anno)]

        datalist = []
        pos_anno = 0
        for _index, (imgline, split) in enumerate(zip(imglist, train_test_split)):
            idx, imgname = [i.strip() for i in imgline.split(' ')]
            idx = int(idx)

            item = {
                'imgname': imgname,
                'imgpath': osp.join(self.img_dir, imgname)
            }

            # landmarks
            landmarks = []
            while (pos_anno<len(annolist)) and (int(annolist[pos_anno][0])==idx):
                anno_i = annolist[pos_anno]
                landmarks.append([float(anno_i[2]), float(anno_i[3])])
                pos_anno += 1
            assert len(landmarks) == 15, f"len = {len(landmarks)} but 15 needed"
            landmarks = np.array(landmarks)
            item['landmarks_raw'] = landmarks

            # split
            if not ((self.split == 'all') or ((self.split=='train')==split)):
                continue
            class_i = int(imgname.split('.')[0].strip())
            
            # select
            if self.select_class is not None:
                if class_i not in self.select_class:
                    continue
            
            # import ipdb; ipdb.set_trace()

            # bbox
            # try:
            bbox_item = bbox_list[_index]
            # except:
            #     import ipdb; ipdb.set_trace()
            bbox_item = np.array([float(itit.strip()) for itit in bbox_item.split(' ')][1:])
            assert bbox_item.shape == (4,), "shape:%d" % (bbox_item.shape)
            bbox_item[2:] = bbox_item[2:] + bbox_item[:2]
            item['bbox'] = bbox_item
            if self.use_crop:
                item['bbox_used'] = bbox_item

            # add to datalist
            datalist.append(item)
        
        return datalist

    def __len__(self):
        if self.dup is None:
            return self.length
        else:
            return self.length * self.dup

    def get_metadata(self, idx):
        return self.datalist[idx % self.length]




