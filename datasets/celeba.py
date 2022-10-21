import json
import os
import random
import os.path as osp

import cv2
import numpy as np
from PIL import Image
from contextlib import contextmanager 
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from utils.soft_points import get_gaussian_map_from_points
from imgtransform.trans_controller import get_trans_controller
from utils.soft_mask import get_smooth_mask

from utils.utils import get_color_index_map, squeeze_recur



class CelebADataset(Dataset):
    def __init__(self, root_dir, 
            select_path=None, 
            anno_path=None, 
            transform=None, 
            choosen_nums=None, choosen_shuffle=False, 
            heatmap_size=None, heatmap_sigma=1, heatmap_mode='ankush',
            pic_trans_num:int=0, pic_trans_type=None, pic_trans_cont=False, 
                pic_return_mask=True, soft_mask=True,
            json_file=None,
            scops_mask_file=None, trans_mask=False, 
            mask_colorize=False, mask_index_list=[2,3,5,6,8],
            predkp_dir=None, saliency_dir=None,
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
        self.scops_mask_file = scops_mask_file
        self.trans_mask = trans_mask
        self.mask_colorize = mask_colorize
        self.mask_index_list = mask_index_list
        self.choosen_nums = choosen_nums
        self.choosen_shuffle = choosen_shuffle

        self.predkp_dir = predkp_dir
        self.saliency_dir = saliency_dir
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
            if not soft_mask:
                self.basemask = torch.ones(128,128).unsqueeze(0)
            else:
                self.basemask = get_smooth_mask(128, 128, 10, 20).unsqueeze(0)

        # print('pic_trans_type:', pic_trans_type)
        if self.pic_trans_num > 0:
            self.trans_controller = get_trans_controller(pic_trans_type)

        self.args = args

        # prepare datalist
        self.datalist = self.preprocess()

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

        if self.json_file is not None:
            jslist = []
            with open(self.json_file, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    jslist.append(item)
                    # js[item['imagename']] = item['landmarks']

        # if self.scops_mask_file is not None:
        #     mask_list = []
        #     with open(self.scops_mask_file, 'r') as f:
        #         for line in tqdm(f, total=len(imglist), postfix='open scops_mask_file'):
        #             item = json.loads(line.strip())
        #             item['mask'] = np.array(item['mask'])
        #             mask_list.append(item)
        #             # js[item['imagename']] = item['landmarks']

        #############################
        # iter
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

            if self.json_file is not None:
                try:
                    assert jslist[idx]['imagename'] == imgname
                except:
                    print('idx:', idx, 'len(jslist):', len(jslist))
                json_item = np.array(jslist[idx]['landmarks'])
                item.update({
                    'json_item': json_item
                })

            # if self.scops_mask_file is not None:
            #     assert mask_list[idx]['imgname'] == imgname
            #     item = mask_list[idx]
            #     item.update({
            #         'scops_mask': item['mask'].transpose(2,0,1)
            #     })

            datalist.append(item)

        # sample if needed
        if self.choosen_nums is not None:
            assert isinstance(self.choosen_nums, int)
            if self.choosen_shuffle:
                ind = random.sample(list(range(len(datalist))), self.choosen_nums)
            else:
                ind = list(range(self.choosen_nums))
            datalist = [datalist[i] for i in ind]


        # print(len(datalist))
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        metadata = self.datalist[idx]
        imgname = osp.basename(metadata['imgname'])
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

        if self.json_file is not None:
            res['json_item'] = metadata['json_item']

        if self.transform is not None:
            res = self.transform(res)

        # heatmap
        if self.heatmap_size is not None:
            heatmap = get_gaussian_map_from_points(landmarks*self.heatmap_size[::-1], hm_size=self.heatmap_size[::-1], sigma=self.heatmap_sigma, mode=self.heatmap_mode)
            res['heatmap'] = heatmap

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

        if self.scops_mask_file is not None:
            scops_mask_path = osp.join(self.scops_mask_file, imgname+'.npy')
            scops_mask_np = np.load(scops_mask_path)
            if scops_mask_np.shape[1] != scops_mask_np.shape[2]: 
                scops_mask_np = scops_mask_np.transpose(2, 0, 1) # 9, 16, 16
            scops_mask = torch.Tensor(scops_mask_np)
            if self.trans_mask:
                scops_mask = self.trans_controller.render(scops_mask.unsqueeze(0), res['img1']['trans_paras'].unsqueeze(0))[0]
            res['scops_mask'] = scops_mask # 9,16,16

            if self.mask_colorize:
                c_idx_map = get_color_index_map(scops_mask_np, self.mask_index_list)
                res['c_idx_map'] = c_idx_map

        if self.predkp_dir is not None:
            predkp_path = osp.join(self.predkp_dir, imgname+'.npy')
            res['predkp'] = np.load(predkp_path)
        
        if self.saliency_dir is not None:
            saliency_path = osp.join(self.saliency_dir, imgname[:-4]+'.png')
            slcy = cv2.imread(saliency_path)[...,::-1]
            slcy = cv2.resize(slcy, (160, 160))[16:16+128, 16:16+128, :]
            idxs = np.where((slcy>=1) & (slcy<=13))
            slcy = rends(idxs[0], idxs[1])
            res['saliency'] = slcy

        return res

    @contextmanager
    def raw_img_pred(self):
        try:
            pic_trans_num_old = self.pic_trans_num
            self.pic_trans_num = 0
            yield self
        finally:
            self.pic_trans_num = pic_trans_num_old

    def update_json_file(self, filename):
        print("Updating json file with %s" % filename)
        self.json_file = filename
        self.datalist = self.preprocess()

        
def rends(idx0, idx1):
    res = torch.zeros((128, 128))
    res[idx0, idx1] = 1
    return res







