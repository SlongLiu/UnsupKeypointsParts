import os
import random
import os.path as osp

import numpy as np
from PIL import Image
from contextlib import contextmanager 

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from utils.soft_points import get_gaussian_map_from_points
from imgtransform.trans_controller import get_trans_controller
from utils.soft_mask import get_smooth_mask

from utils.utils import squeeze_recur



class SimplerCelebA(Dataset):
    def __init__(self, root_dir, 
            select_path=None, 
            anno_path=None, 
            transform=None, 
            choosen_nums=None, choosen_shuffle=False, 
            heatmap_size=None, heatmap_sigma=1, heatmap_mode='ankush',
            pic_trans_num:int=0, pic_trans_type=None, pic_trans_cont=False, pic_return_mask=True,
            json_file=None,
            args=None, **kw):
        """init

        Args:
            root_dir (str): root dir
            select_path (str, optional): the index file of the train/test split. Defaults to None.
            anno_path (str, optional): the anno of each image. Defaults to None.
            transform (transform, optional): transformation each img. Defaults to None.

            choosen_nums (int, optional): number of items used 
            choosen_shuffle (bool): no use if `choosen_nums` is None.

            heatmap_size (tuple(int), optinal): the size of the heatmap if not None
        """
        self.root_dir =  root_dir
        self.select_path = select_path
        self.anno_path = anno_path
        self.transform = transform
        self.json_file = json_file
        # heatmap
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_mode = heatmap_mode
        # pic_trans
        assert isinstance(pic_trans_num, int)
        self.pic_trans_num = pic_trans_num
        self.pic_trans_type = pic_trans_type
        self.pic_trans_cont = pic_trans_cont
        self.pic_return_mask = pic_return_mask
        if self.pic_return_mask:
            self.basemask = get_smooth_mask(128, 128, 10, 20).unsqueeze(0)

        # print('pic_trans_type:', pic_trans_type)
        if self.pic_trans_num > 0:
            self.trans_controller = get_trans_controller(pic_trans_type)

        self.args = args

        # prepare datalist
        self.datalist = self.preprocess()

        if choosen_nums is not None:
            assert isinstance(choosen_nums, int)

            if choosen_shuffle:
                ind = random.sample(list(range(len(self.datalist))), choosen_nums)
            else:
                ind = list(range(choosen_nums))

            self.datalist = [self.datalist[i] for i in ind]
        
        
    def preprocess(self):
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
            annolist = list(range(len(imglist)))
            # raise NotImplementedError("anno must exist for now")

        if self.select_path is not None:
            selectset = set([line.strip() for line in open(self.select_path, 'r')])
        else:
            selectset = None

        for idx, (imgname, anno) in enumerate(zip(imglist, annolist)):
            # skip the imgs not in the selected list
            if selectset is not None and imgname not in selectset:
                continue
            
            # adding to the datalist
            item = {
                'imagepath': osp.join(self.root_dir, imgname),
                'imgname': imgname,
            }
            
            if self.anno_path is not None:
                # parser the anno
                anno = annolist[idx]
                coords = [x.strip() for x in anno.split(' ') if x.strip() != ''][1:]
                coords = np.array([float(x) for x in coords]).reshape(-1,2)
                item.update({'landmarks': coords})

            datalist.append(item)

        # print(len(datalist))
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        metadata = self.datalist[idx]
        res = {}

        # image
        image = Image.open(metadata['imagepath']).convert("RGB")
        img_size = image.size # w,h
        # if self.transform is not None:
        #     image = self.transform(image)
        res['imgname'] = metadata['imgname']
        res['image'] = image

        # landmarks
        if 'landmarks' in metadata:
            landmarks = metadata['landmarks'].copy() # / img_size 
            res['landmarks'] = landmarks

        if self.transform is not None:
            res = self.transform(res)

        # heatmap
        if self.heatmap_size is not None:
            heatmap = get_gaussian_map_from_points(landmarks*self.heatmap_size[::-1], hm_size=self.heatmap_size[::-1], sigma=self.heatmap_sigma, mode=self.heatmap_mode)
            res['heatmap'] = heatmap

        # # pics after transform
        # if self.pic_trans_num > 0:
        #     base_image = res['image']
        #     for i in range(self.pic_trans_num):
        #         output = self.trans_controller(base_image.unsqueeze(0), return_mask=self.pic_return_mask)
        #         new_img = {}
        #         new_img['image_trans'] = output[0].squeeze(0)
        #         if self.pic_return_mask:
        #             new_img['image_mask'] = output[2].squeeze(0)
        #         if output[1] is not None:
        #             new_img['trans_paras'] = squeeze_recur(output[1], 0)
        #         res['img%d' % (i+1)] = new_img
        #         if self.pic_trans_cont:
        #             base_image = new_img['image_trans']

        # pics after transform
        if self.pic_trans_num > 0:
            base_image = res['image']
            for i in range(self.pic_trans_num):
                new_img = {}
                if self.pic_return_mask:
                    image_all = torch.cat((base_image.unsqueeze(0), self.basemask.unsqueeze(0)), dim=1)
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

    @contextmanager
    def raw_img_pred(self):
        try:
            pic_trans_num_old = self.pic_trans_num
            self.pic_trans_num = 0
            yield self
        finally:
            self.pic_trans_num = pic_trans_num_old

        








