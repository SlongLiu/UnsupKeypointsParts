import os, sys
import os.path as osp
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from datasets.basedataset import BaseDataSet

# 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 
# 6=bus, 7=car , 8=cat, 9=chair, 10=cow, 
# 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 
# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

class PascalVoc(BaseDataSet, Dataset):
    def __init__(self,
        root_dir,
        classselect,
        transform,
        dup = None,

        pic_trans_num:int=0, pic_trans_type=None, pic_trans_cont=False, 
            pic_return_mask=False, soft_mask=True,
        args=None,
        **kw,
    ):
        self.root_dir = root_dir
        self.classselect = classselect
        self.transform = transform
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
        listtxt = osp.join(self.root_dir, 'ImageSets/SelectSeg3', f'list_{self.classselect}.txt')
        imgnamelist = [x.strip() for x in open(listtxt, 'r')]

        datalist = []
        for idx, imgname in enumerate(imgnamelist):
            imgpath = osp.join(self.root_dir, 'JPEGImages', imgname+'.jpg')
            xmlpath = osp.join(self.root_dir, 'Annotations', imgname+'.xml')
            segmappath = osp.join(self.root_dir, 'SegmentationClass', imgname+'.png')
            item = {
                'imgname': imgname+'.jpg',
                'imgpath': imgpath,
                'segmappath': segmappath,
            }

            # parse xml
            tree = ET.parse(xmlpath)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            item['imgsize'] = np.array([width, height]).astype(float)

            oblist = root.findall('object')
            for ob in oblist:
                obname = ob.find('name').text
                if obname.strip() != self.classselect:
                    continue
                truncated = int(ob.find('truncated').text)
                if truncated:
                    continue
                bndbox = ob.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                w = xmax - xmin
                h = ymax - ymin
                if w*h / (width * height) <= 0.2:
                    continue
                # add object to item dict
                item['bbox'] = np.array([xmin, ymin, xmax, ymax])
                ratio = 0.3
                item['bbox_used'] = np.array([xmin - w*ratio, ymin - h*ratio, xmax + w*ratio, ymax + h*ratio])
                break
        
            datalist.append(item)
        return datalist

    def __len__(self) -> int:
        if self.dup is None:
            return self.length
        else:
            return self.length * self.dup

    def get_metadata(self, idx):
        return self.datalist[idx % self.length]


