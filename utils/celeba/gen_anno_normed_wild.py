import os
from os import replace
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm

def main():
    filepath = '/data/shilong/data/wildceleba/anno/list_landmarks_celeba.txt'
    imgdir = '/data/shilong/data/wildceleba/Img'
    outfilepath = '/data/shilong/data/wildceleba/anno/list_landmarks_celeba_normed2.txt'

    with open(filepath, 'r') as fin:
        with open(outfilepath, 'w') as fout:
            fout.write(fin.readline())
            fout.write(fin.readline())

            # read lm
            for line in tqdm(fin):
                # parser the anno
                parserlist = [x.strip() for x in line.strip().replace('\t', ' ').split(' ') if x.strip() != '']
                filename = parserlist[0].strip()
                filepath = osp.join(imgdir, filename)
                img = cv2.imread(filepath)[...,::-1]
                H,W,_ = img.shape
                coords = parserlist[1:]
                coords = np.array([float(x) for x in coords]).reshape(-1,2)
                coords = coords / (W, H)
                text = filename + ' ' + ' '.join([str(x) for x in coords.reshape(-1).tolist()]) + '\n'
                fout.write(text)


if __name__ == "__main__":
    main()