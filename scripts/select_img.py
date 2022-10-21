import os, sys
from os.path import dirname
import os.path as osp
sys.path.append(osp.dirname(sys.path[0]))

import matplotlib.pyplot as plt
import cv2

thedir = 'tmp/new_rank_plot/l2_cl_model_after_relu5_2'

def main():
    for i in range(10):
        dirname = osp.join(thedir, 'feat%d' % i)
        row = 4
        col = 5
        plt.figure(figsize=(col*3, row*3), dpi=100)
        imglist = sorted(os.listdir(dirname), reverse=True)
        # most relavant
        for j in range(10):
            imgname = osp.join(dirname, imglist[j])
            ax = plt.subplot(row, col, j+1)
            # img = cv2.imread(imgname)[...,::-1]
            # print(img.shape)
            # raise ValueError
            ax.imshow(cv2.imread(imgname)[...,::-1][40:440, 100:540, :])
            ax.set_axis_off()

            score = imglist[j].split('-')[-1].strip()[:-4]
            ax.set_title("score:"+score)

        for j in range(10):
            k = len(imglist) - j - 1
            imgname = osp.join(dirname, imglist[k])
            ax = plt.subplot(row, col, 10 + j + 1)
            ax.imshow(cv2.imread(imgname)[...,::-1][40:440, 100:540, :])
            ax.set_axis_off()

            score = imglist[k].split('-')[-1].strip()[:-4]
            ax.set_title("score:"+score)

        os.makedirs(osp.join(thedir, 'all'), exist_ok=True)
        plt.savefig(osp.join(thedir, 'all/feat-%d' % i))
        plt.close()




if __name__ == "__main__":
    main()