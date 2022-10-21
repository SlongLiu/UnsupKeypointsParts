import os, sys
import os.path as osp
sys.path.append(osp.dirname(sys.path[0]))
from tqdm import tqdm
import scipy.io as sio
import numpy as np

from datasets.penn_action import PennAction

root_dir='/data/shilong/data/penn_action/Penn_Action'
output_dir = '/data/shilong/data/penn_action/Penn_Action_CropNoScale'
savedir = osp.join(output_dir, 'frames')
cropinfo_savedir = osp.join(output_dir, 'cropbboxs')
os.makedirs(savedir, exist_ok=True)
os.makedirs(cropinfo_savedir, exist_ok=True)
ds = PennAction(
    root_dir=root_dir,
    cropped=False,
)

for i in tqdm(range(len(ds))):
    sample = ds[i]
    
    # crop img
    x0,y0,x1,y1 = sample['bbox']
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    # lengthx = lengthy = max(x1-x0, y1-y0) * 1.1
    deltap = (y1 - y0) * 0.2
    lengthx = (x1 - x0) + deltap
    lengthy = (y1 - y0) + deltap
    xmin, xmax = xm - lengthx/2, xm + lengthx/2
    ymin, ymax = ym - lengthy/2, ym + lengthy/2
    img = sample['image'].crop((xmin, ymin, xmax, ymax))

    # savepath
    savepath = osp.join(savedir, sample['imgname'])
    os.makedirs(osp.dirname(savepath), exist_ok=True)
    try:
        img.save(savepath)
    except Exception as e:
        print("Error in saving %s" % savepath)
        print("Error info: %s" % e)
        continue

    # save cropbbox
    cropbbox_savepath = osp.join(cropinfo_savedir, osp.splitext(sample['imgname'])[0]+'.npy')
    os.makedirs(osp.dirname(cropbbox_savepath), exist_ok=True)
    np.save(cropbbox_savepath, np.array([xmin, ymin, xmax, ymax]))




    


