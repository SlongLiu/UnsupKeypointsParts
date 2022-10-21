import os, sys
import os.path as osp
sys.path.append(osp.dirname(sys.path[0]))

from tqdm import tqdm
import cv2
from PIL import Image

import torchvision.transforms as T

from datasets.wildceleba import WildCelebA

# img_out_dir = '/data/shilong/data/wildceleba/Img_crop'
list_path = '/data/shilong/data/wildceleba/MAFL_useful/testing.txt'


ds = WildCelebA(
    root_dir='/data/shilong/data/wildceleba/Img',
    select_path='/data/shilong/data/wildceleba/MAFL/testing.txt',
    anno_path='/data/shilong/data/wildceleba/anno/list_landmarks_celeba.txt',
    bbox_path='/data/shilong/data/wildceleba/anno/list_bbox_celeba.txt',
    transform=T.ToTensor(),
)

cnt = 0
for i in tqdm(range(len(ds))):
    sample = ds[i]
    img = sample['image']
    _, H, W = img.shape
    area = H * W
    bbox = sample['bbox']
    area_bbox = bbox[1,0] * bbox[1,1]
    ratio = area_bbox / area
    if ratio > 0.3:
        cnt += 1
        with open(list_path, 'a') as f:
            f.write(osp.basename(sample['imgname']) + '\n')

print("total image used: %d" % cnt)

