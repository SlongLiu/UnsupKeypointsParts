import os, sys
import os.path as osp
import random

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import cv2
from PIL import Image
import copy

from torch.utils.data import Dataset
from datasets.basedataset import BaseDataSet

# train: 87542
# test: 76297
PENN_DICR = {'baseball_pitch': 167, 
            'baseball_swing': 173, 
            'bench_press': 140, 
            'bowl': 220, 
            'clean_and_jerk': 88, 
            'golf_swing': 166, 
            'jump_rope': 82, 
            'jumping_jacks': 112, 
            'pullup': 199, 
            'pushup': 211, 
            'situp': 100, 
            'squat': 231, 
            'strum_guitar': 94, 
            'tennis_forehand': 157, 
            'tennis_serve': 186}


class PennAction(BaseDataSet, Dataset):
    def __init__(self,
        root_dir = '',
        frames_each_video = None,
        split = 'all', # 
        cropped = True,
        select_class = None,
        dup=None,

        transform=None,
        pic_trans_num:int=0, pic_trans_type=None, pic_trans_cont=False, 
                pic_return_mask=False, soft_mask=True,
        args=None,
        **kw):
        self.root_dir = root_dir
        self.frames_each_video = frames_each_video
        self.split = split
        assert split in ['train', 'test', 'all']
        self.cropped = cropped
        self.select_class = select_class
        if self.select_class is not None and isinstance(self.select_class, str):
            self.select_class = [self.select_class]
        self.dup = dup

        self.transform = transform
        self.pic_trans_num = pic_trans_num
        self.pic_trans_type = pic_trans_type
        self.pic_trans_cont = pic_trans_cont
        self.pic_return_mask = pic_return_mask
        self.soft_mask = soft_mask

        self.datalist = self.preprocess()
        self.length = len(self.datalist)
        self.post_process()

    def preprocess(self):
        data_dir = osp.join(self.root_dir, 'frames')
        videolist = sorted(os.listdir(data_dir))
        datalist = []

        for videoname in videolist:
            # video anno
            annopath = osp.join(self.root_dir, 'labels', videoname+'.mat')
            anno = sio.loadmat(annopath)
            videoclass = anno['action'].item()
            if self.select_class is not None and (videoclass not in self.select_class):
                continue

            x_all = anno['x']
            y_all = anno['y']
            bbox_all = anno['bbox']
            vis_all = anno['visibility']
            is_train = anno['train'].item()
            if self.split == 'all' or (self.split == 'train' and is_train == 1) or (self.split == 'test' and is_train != 1):
                pass
            else:
                continue
            
            # each frame
            # print(osp.join(data_dir, videoname))
            framelist = sorted(os.listdir(osp.join(data_dir, videoname)))
            if self.frames_each_video is not None:
                if self.frames_each_video <= len(framelist):
                    framelist = random.sample(framelist, self.frames_each_video)
                else:
                    framelist = [random.choice(framelist) for i in range(self.frames_each_video)]
            for framename in framelist:
                frame_id = int(osp.splitext(framename)[0]) - 1
                item = {
                    'imgname': osp.join(videoname, framename),
                    'imgpath': osp.join(data_dir, videoname, framename)
                }

                # cropbox
                if self.cropped:
                    cboxpath = osp.join(self.root_dir, 'cropbboxs', videoname, osp.splitext(framename)[0]+'.npy')
                    if not osp.exists(cboxpath):
                        print("Unfound %s" % cboxpath)
                        continue
                    cbox = np.load(cboxpath)
                    xmin, ymin, xmax, ymax = cbox

                # landmarks
                landmarks_raw = np.concatenate((x_all[frame_id], y_all[frame_id])).reshape(2,-1).transpose(1,0).copy()
                assert landmarks_raw.shape == (13, 2)
                if self.cropped:
                    landmarks_raw = landmarks_raw - (xmin, ymin)
                item['landmarks_raw'] = landmarks_raw.astype(float)


                # visibility
                visibility = vis_all[frame_id]
                item['visibility'] = visibility.astype(float)

                # bbox
                try:
                    bbox = bbox_all[frame_id] # x0, y0, x1, y1
                except:
                    print(f"No bbox data in {videoname}/{frame_id}")
                    continue
                if self.cropped:
                    bbox = bbox.copy() - (xmin, ymin, xmin, ymin)
                item['bbox'] = bbox.astype(float)

                # add to datalist
                datalist.append(item)

        return datalist

    def __len__(self) -> int:
        if self.dup is not None:
            return self.length * self.dup
        else:
            return self.length

    def get_metadata(self, idx):
        return self.datalist[idx % self.length]

# PennActionDouble

class PennActionDouble(Dataset):
    def __init__(self,
        root_dir = '',
        frames_each_video = None,
        split = 'all', # 
        cropped = True,
        select_class = None,
        dup = None,

        transform=None,
        args=None,
        **kw):
        self.root_dir = root_dir
        self.frames_each_video = frames_each_video
        self.split = split
        assert split in ['train', 'test', 'all']
        self.cropped = cropped
        self.select_class = select_class
        if self.select_class is not None and isinstance(self.select_class, str):
            self.select_class = [self.select_class]

        self.transform = transform

        self.datalist = self.preprocess()
        self.length = len(self.datalist)

    def preprocess(self):
        data_dir = osp.join(self.root_dir, 'frames')
        videolist = sorted(os.listdir(data_dir))
        datalist = []

        for videoname in videolist:
            # video anno
            annopath = osp.join(self.root_dir, 'labels', videoname+'.mat')
            anno = sio.loadmat(annopath)
            videoclass = anno['action'].item()
            if self.select_class is not None and (videoclass not in self.select_class):
                continue

            x_all = anno['x']
            y_all = anno['y']
            bbox_all = anno['bbox']
            vis_all = anno['visibility']
            is_train = anno['train'].item()
            if self.split == 'all' or (self.split == 'train' and is_train == 1) or (self.split == 'test' and is_train != 1):
                pass
            else:
                continue
            
            # each frame
            # print(osp.join(data_dir, videoname))
            framelist = sorted(os.listdir(osp.join(data_dir, videoname)))
            if self.frames_each_video is not None:
                framelist1 = [random.choice(framelist) for i in range(self.frames_each_video)]
                framelist2 = [random.choice(framelist) for i in range(self.frames_each_video)]
            else:
                framelist1 = random.shuffle(framelist)
                framelist2 = random.shuffle(framelist)

            for framename1, framename2 in zip(framelist1, framelist2):
                double_item = []
                for framename in (framename1, framename2):
                    frame_id = int(osp.splitext(framename)[0]) - 1
                    item = {
                        'imgname': osp.join(videoname, framename),
                        'imgpath': osp.join(data_dir, videoname, framename)
                    }

                    # cropbox
                    if self.cropped:
                        cboxpath = osp.join(self.root_dir, 'cropbboxs', videoname, osp.splitext(framename)[0]+'.npy')
                        if not osp.exists(cboxpath):
                            print("Unfound %s" % cboxpath)
                            double_item.append(None)
                            continue
                        cbox = np.load(cboxpath)
                        xmin, ymin, xmax, ymax = cbox

                    # landmarks
                    landmarks_raw = np.concatenate((x_all[frame_id], y_all[frame_id])).reshape(2,-1).transpose(1,0)
                    assert landmarks_raw.shape == (13, 2)
                    if self.cropped:
                        landmarks_raw = landmarks_raw - (xmin, ymin)
                    item['landmarks_raw'] = landmarks_raw.astype(float)

                    # visibility
                    visibility = vis_all[frame_id]
                    item['visibility'] = visibility.astype(float)

                    # bbox
                    try:
                        bbox = bbox_all[frame_id] # x0, y0, x1, y1
                    except:
                        print(f"No bbox data in {videoname}/{frame_id}")
                        double_item.append(None)
                        continue
                    if self.cropped:
                        bbox = bbox.copy() - (xmin, ymin, xmin, ymin)
                    item['bbox'] = bbox.astype(float)

                    double_item.append(item)

                # add to datalist
                if None in double_item:
                    continue
                if double_item[0]['imgname'] == double_item[1]['imgname']:
                    continue
                datalist.append(double_item)

        return datalist

    def __len__(self) -> int:
        if self.dup is not None:
            return self.length * self.dup
        else:
            return self.length

    def get_metadata(self, idx):
        return self.datalist[idx % self.length]

    def __getitem__(self, idx: int):
        """
            metadata: "imgpath" and "imgname"
        """
        metadata = self.get_metadata(idx)
        res = {
            'img1': {},
            'img2': {}
        }

        # image
        image1 = Image.open(metadata[0]['imgpath']).convert("RGB")
        if self.transform is not None:
            image1 = self.transform(image1)
        res['img1']['image_trans'] =  image1

        image2 = Image.open(metadata[1]['imgpath']).convert("RGB")
        if self.transform is not None:
            image2 = self.transform(image2)
        res['img2']['image_trans'] =  image2

        h,w,c = image1.shape
        # mask
        mask1 = np.ones((h,w,c))
        res['img1']['image_mask'] = mask1
        mask2 = np.ones((h,w,c))
        res['img2']['image_mask'] = mask2
        
        return res
                






