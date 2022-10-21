import os, sys
import numpy as np
import os.path as osp

from tqdm import tqdm

def main():
    imgdir = '/data/shilong/data/imm/aflw_sl/Image'
    traintxt = '/data/shilong/data/imm/aflw_sl/split/train.txt'
    testtxt = '/data/shilong/data/imm/aflw_sl/split/test.txt'
    annopath = '/data/shilong/data/imm/aflw_sl/anno/anno.txt'
    os.makedirs('tmp/AFLW_new', exist_ok=True)

    annodict = {}

    with open(annopath, 'r') as f:
        f.readline()
        f.readline()
        for line in f:
            itemlist = [x.strip() for x in line.split(' ') if x.strip() != '']
            coords = np.array([float(x) for x in itemlist[1:]]).reshape(-1,2)
            annodict[itemlist[0]] = coords[np.newaxis, ...]

    print("save train")
    coords_train = None
    for line in tqdm(open(traintxt, 'r')):
        imgname = line.strip()
        if coords_train is None:
            coords_train = annodict[imgname]
        else:
            coords_train = np.concatenate((coords_train, annodict[imgname]), 0)
    print('coords_train.shape:', coords_train.shape)
    np.save('tmp/AFLW_new/training.npy', coords_train)

    coords_test = None
    for line in tqdm(open(testtxt, 'r')):
        imgname = line.strip()
        if coords_test is None:
            coords_test = annodict[imgname]
        else:
            coords_test = np.concatenate((coords_test, annodict[imgname]), 0)
    print('coords_test.shape:', coords_test.shape)
    np.save('tmp/AFLW_new/testing.npy', coords_test)



if __name__ == "__main__":
    main()