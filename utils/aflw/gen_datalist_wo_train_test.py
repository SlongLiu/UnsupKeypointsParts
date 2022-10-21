import os
# from os import replace
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm


root_dir = '/data/shilong/data/AFLW/wild_aflw/aflw/data/flickr'
dirlist = os.listdir(root_dir)
imglist = []
for dirname in dirlist:
    imglist.extend([osp.join(dirname, i) for i in os.listdir(osp.join(root_dir, dirname))])
imglist.sort()

for imgname in tqdm(imglist):
    