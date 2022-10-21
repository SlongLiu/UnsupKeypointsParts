import os, sys
import os.path as osp
sys.path.append(osp.dirname(sys.path[0]))

from datasets.get_loader import get_dataset_by_config
from utils.slconfig import SLConfig
import matplotlib.pyplot as plt


def main():
    args = SLConfig.fromfile('configs/IMM/MyIMM_CELEBA_v3_128x128_TPS_PERCEP_10lm_h32_newtest.py')
    train_set, test_set = get_dataset_by_config(args)

    n = 4
    row = 4
    col = 5
    plt.figure(figsize=(col * 3, row * 3), dpi=100)
    for i in range(n):
        sample = train_set[0]
        image = sample['image'].permute(1,2,0)
        img1 = sample['img1']['image_trans'].permute(1,2,0)
        img1_mask = sample['img1']['image_mask'].permute(1,2,0)
        para1 = sample['img1']['trans_paras']

        img2 = sample['img2']['image_trans'].permute(1,2,0)
        img2_mask = sample['img2']['image_mask'].permute(1,2,0)
        para2 = sample['img2']['trans_paras']

        # print('para1:', para1)
        # print('para2:', para2)

        ax1 = plt.subplot(row, col, i*col + 1)
        ax1.imshow(image)
        ax1.set_axis_off()

        ax2 = plt.subplot(row, col, i*col + 2)
        ax2.imshow(img1)
        ax2.set_axis_off()

        ax3 = plt.subplot(row, col, i*col + 3)
        ax3.imshow(img1_mask, cmap='gray')
        ax3.set_axis_off()

        ax4 = plt.subplot(row, col, i*col + 4)
        ax4.imshow(img2)
        ax4.set_axis_off()

        ax5 = plt.subplot(row, col, i*col + 5)
        ax5.imshow(img2_mask, cmap='gray')
        ax5.set_axis_off()

    plt.savefig('tmp/tps.para.testv2_v0.3_5kp_nopad.png')
        

if __name__ == "__main__":
    main()