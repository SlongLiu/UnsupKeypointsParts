import os
import os.path as osp

from PIL import Image
import numpy as np
from torch import imag
from tqdm import tqdm

imgfolder = '/data/shilong/data/imm/celeba/Img/img_align_celeba_hq'
annopath = '/data/shilong/data/imm/celeba/Anno/list_landmarks_align_celeba.txt'
outputfile = '/data/shilong/data/imm/celeba/Anno/list_img_size_celeba.txt'

def main():
    # open anno_path and select_path
    annolist = [line.strip() for line in open(annopath, 'r')]
    annolist = annolist[2:]

    # preprocess
    # with open(outputfile, 'w') as f:
    #     f.write("%d\n" % len(annolist))
    #     f.write("width height\n")
    if 1:
        for idx, anno in enumerate(tqdm(annolist)):
            # parser the anno
            anno = annolist[idx]
            line = [x.strip() for x in anno.split(' ') if x.strip() != '']
            imgname = line[0]
            coords = line[1:]
            coords = np.array([int(x) for x in coords]).reshape(-1,2)
            
            # get img size
            imgpath = osp.join(imgfolder, imgname)
            img = Image.open(imgpath)
            width, height = img.size
            if width != 178 or height != 218:
                raise ValueError("width=%d height=%d" % (width, height))

            # f.write("%s %d %d\n" % (imgname, width, height))


if __name__ == "__main__":
    main()