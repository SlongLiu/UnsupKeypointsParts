
import os, sys
import os.path as osp

sys.path.append(osp.dirname(sys.path[0]))

import matplotlib.pyplot as plt


from utils.slconfig import SLConfig
from datasets.get_loader import get_dataset_by_config
from utils.plot_landmarks import plot_landmarks



def main():
    args = SLConfig.fromfile('/data/shilong/code/slimm/configs/IMM/MyIMM_CELEBA_v2_128x128_TPS_PERCEP_10lm_h32.py')
    trainset, testset = get_dataset_by_config(args)

    print(len(trainset))
    sample = trainset[0]
    image, lm = sample['image'], sample['landmarks']

    # plt.figure(figsize=(10, 10), dpi=120)
    ax1 = plt.subplot(111)
    ax1.imshow(image.permute(1,2,0))
    plot_landmarks(ax1, lm*(128, 128))

    plt.savefig('tmp/transform_test.png')



if __name__ == "__main__":
    main()
