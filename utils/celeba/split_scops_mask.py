import os
import json
import numpy as np
from tqdm import tqdm


INPUT = '/data/shilong/data/imm/celeba/mask/mask_45000.json'
OUTPUTDIR = '/data/shilong/data/imm/celeba/scops_mask'
os.makedirs(OUTPUTDIR, exist_ok=True)

with open(INPUT, 'r') as f:
    for line in tqdm(f, total=202599, postfix='open scops_mask_file'):
        item = json.loads(line.strip())
        scops_mask = np.array(item['mask'])
        imgname = item['imgname']
        np.save(os.path.join(OUTPUTDIR, imgname), scops_mask)

