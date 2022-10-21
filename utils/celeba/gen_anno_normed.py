import os
from os import replace
import os.path
from typing import Text
import numpy as np

filepath = '/data/shilong/data/imm/celeba/Anno/list_landmarks_align_celeba.txt'
outfilepath = '/data/shilong/data/imm/celeba/Anno/list_landmarks_align_celeba_normed.txt'

def main():
    with open(filepath, 'r') as fin:
        with open(outfilepath, 'w') as fout:
            fout.write(fin.readline())
            fout.write(fin.readline())

            # read lm
            for line in fin:
                # parser the anno
                parserlist = [x.strip() for x in line.strip().replace('\t', ' ').split(' ') if x.strip() != '']
                filename = parserlist[0].strip()
                coords = parserlist[1:]
                coords = np.array([float(x) for x in coords]).reshape(-1,2)
                coords = coords / (178, 218)
                text = filename + ' ' + ' '.join([str(x) for x in coords.reshape(-1).tolist()]) + '\n'
                fout.write(text)


if __name__ == "__main__":
    main()