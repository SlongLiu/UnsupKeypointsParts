import os, sys
import os.path as osp
from typing import Dict

sys.path.append(osp.dirname(sys.path[0]))

import torchvision.transforms as T
from addict import Dict
from datasets.wildceleba import WildCelebA
from utils.utils import slprint

TRANS_CONTROLLER = dict(
    NAME = 'randTPStransform2',
    PARAS = dict(
        height = 128,
        width = 128,
    )
)
TRANS_CONTROLLER = Dict(TRANS_CONTROLLER)

ds = WildCelebA(
    root_dir='/data/shilong/data/wildceleba/Img',
    select_path='/data/shilong/data/wildceleba/MAFL/training.txt',
    anno_path='/data/shilong/data/wildceleba/anno/list_landmarks_celeba.txt',
    bbox_path='/data/shilong/data/wildceleba/anno/list_bbox_celeba.txt',
    transform=T.Compose([T.Resize((128, 128)), T.ToTensor()]),
    saliency_dir='/data/shilong/data/wildceleba/Saliency_Wild',
    pic_trans_num = 2, pic_trans_type = TRANS_CONTROLLER, pic_return_mask = True, pic_trans_cont=False, soft_mask=False,
)

sample = ds[0]
slprint(sample, 'sample')

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10), dpi=200)
ax1 = plt.subplot(221)
plt.imshow(sample['img1']['image_trans'].permute(1,2,0))

ax2 = plt.subplot(222)
plt.imshow(sample['img1']['image_mask'][0], cmap='gray')

ax1 = plt.subplot(223)
plt.imshow(sample['img2']['image_trans'].permute(1,2,0))

ax2 = plt.subplot(224)
plt.imshow(sample['img2']['image_mask'][0], cmap='gray')


plt.savefig('tmp/wildceleba_saliency.png')

