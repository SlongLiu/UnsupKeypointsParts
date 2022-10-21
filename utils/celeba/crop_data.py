import os, sys
import os.path as osp

import numpy as np
import cv2
from tqdm import tqdm

class AnnoViewer:
    def __init__(self, annopath):
        self.annodict = {}

        annolist = [line.strip() for line in open(annopath, 'r')]
        annolist = annolist[2:]
        
        for anno in annolist:
            items = [x.strip() for x in anno.split(' ') if x.strip() != '']
            coords = items[1:]
            coords = np.array([float(x) for x in coords]).reshape(-1,2)
            filename = items[0]
            self.annodict.update({filename: coords})
                
    def __getitem__(self, filename):
        return self.annodict[filename]


class MyCenterCrop():
    def __init__(self, outshape=(160, 160)):
        self.outshape = outshape

    def __call__(self, img, anno):
        """

        1. center crop 160x160
        # 2. resize to 128x128

        Args:
            img (np.array(H,W,3)): [description]
            anno (np.array(N,2)): range(1...W, 1...H)

        Output:
            img_new: 
            anno_new:
        """
        H, W, _ = img.shape
        oH, oW = self.outshape
        ulx = int((W - oW) / 2)
        uly = int((H - oH) / 2)
        drx = ulx + oW
        dry = uly + oH

        img_new = img[uly:dry, ulx:drx, :]
        anno_new = anno - (ulx, uly)
        anno_new = anno_new / img_new.shape[:2][::-1]

        return img_new, anno_new

class annofile:
    def __init__(self, path):
        self.path = path
        self.itemdict = {} # filename: annotext

    def addline(self, filename, text):
        self.itemdict.update({filename: text})

    def dump(self):
        itemlist = sorted(self.itemdict.keys())
        totalnumb = len(itemlist)
        with open(self.path, 'w') as f:
            f.write("%d\n" % totalnumb)
            f.write('lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y\n')

            for item in itemlist:
                fulltext = item + ' ' + self.itemdict[item] + '\n'
                f.write(fulltext)

def main():
    # path
    dirpath = '/data/shilong/data/imm/celeba/Img/img_align_celeba_hq'
    anno_path = '/data/shilong/data/imm/celeba/Anno/list_landmarks_align_celeba_normed.txt'
    av = AnnoViewer(anno_path)
    cf = MyCenterCrop()
    

    # output
    outputdir = '/data/shilong/data/celeba_sl/Img/img_align_celeba_hq'
    annooutpath = '/data/shilong/data/celeba_sl/Anno/list_landmarks_align_celeba_normed.txt'
    os.makedirs(outputdir, exist_ok=True)
    os.makedirs(osp.dirname(annooutpath), exist_ok=True)
    af = annofile(annooutpath)

    filelist = sorted([osp.join(dirpath, x) for x in os.listdir(dirpath)])
    for filepath in tqdm(filelist):
        filename = osp.basename(filepath)
        img = cv2.imread(filepath)[...,::-1]
        anno = av[filename] * img.shape[:2][::-1]
        
        img_new, anno_new = cf(img, anno)

        # write to file
        cv2.imwrite(osp.join(outputdir, filename), img_new[...,::-1])
        anno_new = anno_new.reshape(-1).tolist()
        text = ' '.join([str(x) for x in anno_new])
        af.addline(filename, text)
    
    af.dump()



if __name__ == "__main__":
    main()