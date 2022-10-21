import os, sys
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
import cv2

from torch.utils.data import Dataset
import torchvision.transforms.functional as F
# from abc import ABC

from utils.soft_mask import get_smooth_mask
from imgtransform.trans_controller import get_trans_controller
from utils.utils import squeeze_recur
from utils.soft_points import get_gaussian_map_from_points

import copy

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataSet():
    def __init__(self):
        raise NotImplementedError

    def post_process(self):
        self.check_attr()
        if self.pic_trans_num > 0:
            self.trans_controller = get_trans_controller(self.pic_trans_type)

        h = self.pic_trans_type.PARAS.height
        w = self.pic_trans_type.PARAS.width

        if self.pic_return_mask:
            if not self.soft_mask:
                self.basemask = torch.ones(h, w).unsqueeze(0)
            else:
                self.basemask = get_smooth_mask(h , w, int(10*h/128), int(20*h/128)).unsqueeze(0)

    def get_metadata(self, idx):
        raise NotImplementedError
        # pass

    def check_attr(self):
        assert hasattr(self, 'transform')
        # assert hasattr(self, 'heatmap_size')
        # assert hasattr(self, 'heatmap_sigma')
        # assert hasattr(self, 'heatmap_mode')

        assert hasattr(self, 'pic_trans_num')
        assert hasattr(self, 'pic_trans_type')
        assert hasattr(self, 'pic_trans_cont')
        assert hasattr(self, 'pic_return_mask')
        assert hasattr(self, 'soft_mask')

        # assert hasattr(self, 'trans_controller')


    def __getitem__(self, idx: int):
        """
            metadata: "imgpath" and "imgname"
        """
        metadata = self.get_metadata(idx)
        imgname = osp.basename(metadata['imgname'])
        res = copy.deepcopy(metadata)

        # image
        try:
            image = Image.open(metadata['imgpath']).convert("RGB")
        except Exception as e:
            print(e)
            print('imgpath:', metadata['imgpath'])
            raise ValueError
        
        img_size_before_crop = image.size
        # landmarks
        if 'landmarks' in metadata:
            landmarks = metadata['landmarks'].copy() # / img_size 
            res['landmarks'] = landmarks
        elif 'landmarks_raw' in metadata:
            landmarks = metadata['landmarks_raw'].copy() / img_size_before_crop 
            res['landmarks'] = landmarks
        else:
            pass
            # raise ValueError('landmarks_raw or landmarks must in metadata')
            # assert 'segmappath' in metadata, ('landmarks_raw or landmarks or segmappath must in metadata')

        # crop the image if need
        if 'bbox_used' in metadata:
            # image_raw = np.array(image)
            # res['image_raw'] = image_raw
            # print(metadata['bbox_used'])
            # import ipdb; ipdb.set_trace()
            image = image.crop(metadata['bbox_used'])
            if 'landmarks' in res:
                _landmarks = (res['landmarks'] * img_size_before_crop - metadata['bbox_used'][:2])
                res['landmarks'] = _landmarks / image.size

        # raw_img_size = image.size
        # res['raw_img_size'] = np.array(image.size)

        img_size = image.size # w,h
        if self.transform is not None:
            image = self.transform(image)
            # image_raw = self.transform(image_raw)
        res['image'] = image
        res['imgsize'] = np.array(img_size)

        # if 'segmappath' in metadata:
        #     segmap = cv2.imread(metadata['segmappath'])[...,::-1]
        #     # if self.transform is not None:
        #     #     segmap = Image.fromarray(segmap)
        #     #     segmap = self.transform(segmap)
        #     res['segmap'] = segmap

        # if self.transform is not None:
        #     res = self.transform(res)

        the_mask = self.basemask
        if 'saliency_path' in metadata:
            saliency_map = Image.open(metadata['saliency_path'][:-4]+'.png')
            # import ipdb; ipdb.set_trace()
            saliency_map = saliency_map.resize((image.shape[2], image.shape[1]))
            # print("saliency_map.shape:", saliency_map.size,  img_size)
            saliency_map = torch.Tensor(np.array((saliency_map)))/255
            res['saliency_map'] = saliency_map.clone()
            # print("saliency_map.shape:", saliency_map.shape)        
            the_mask = saliency_map * the_mask
            # import ipdb; ipdb.set_trace()

        # pics after transform
        if self.pic_trans_num > 0:
            base_image = res['image']
            for i in range(self.pic_trans_num):
                new_img = {}
                if self.pic_return_mask:
                    image_all = torch.cat((base_image.unsqueeze(0), the_mask.unsqueeze(0)), dim=1)
                    output = self.trans_controller(image_all, return_mask=False)
                    new_img['image_trans'] = output[0].squeeze(0)[:3,...]   # 3,H,W
                    new_img['image_mask'] = output[0].squeeze(0)[3:,...]    # 1,H,W
                else:
                    output = self.trans_controller(base_image.unsqueeze(0), return_mask=False)
                    new_img['image_trans'] = output[0].squeeze(0)
                if output[1] is not None:
                    new_img['trans_paras'] = squeeze_recur(output[1], 0)
                res['img%d' % (i+1)] = new_img
                if self.pic_trans_cont:
                    base_image = new_img['image_trans']

        return res



