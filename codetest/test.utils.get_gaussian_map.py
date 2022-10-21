import os, sys
import os.path as osp
sys.path.append(osp.dirname(sys.path[0]))

import torch
import matplotlib.pyplot as plt

from utils.soft_points import get_gaussian_map_from_points, get_expected_points_from_map

def main():
    points = torch.rand(1, 1, 2)
    # points = points
    print(points)
    std = 0.02

    plt.figure(figsize=(5,15), dpi=80) # 3x1
    
    ax1 = plt.subplot(411)
    map1 = get_gaussian_map_from_points(points, height=80, weight=120, std=std, mode='rot')[0,0]
    ax1.set_title("rot mode")
    ax1.set_axis_off()
    ax1.imshow(map1, cmap='gray')

    ax2 = plt.subplot(412)
    map2 = get_gaussian_map_from_points(points, height=80, weight=120, std=std, mode='flat')[0,0]
    ax2.set_title("flat mode")
    ax2.set_axis_off()
    ax2.imshow(map2, cmap='gray')

    ax3 = plt.subplot(413)
    map3 = get_gaussian_map_from_points(points, height=80, weight=120, std=std, mode='ankush')[0,0]
    ax3.set_title("ankush mode")
    ax3.set_axis_off()
    ax3.imshow(map3, cmap='gray')

    ax4 = plt.subplot(414)
    map4 = get_gaussian_map_from_points(points, height=80, weight=120, std=std, mode='point')[0,0]
    ax4.set_title("point mode")
    ax4.set_axis_off()
    ax4.imshow(map4, cmap='gray')

    plt.savefig('gaussian_0.02.png')
    print('points:', points)

    print('=' * 20)
    for map_ in [map1, map2, map3, map4]:
        print('max min:', map_.max(), map_.min())
        print('expect:', get_expected_points_from_map(10*map_.unsqueeze(0).unsqueeze(0)))


if __name__ == "__main__":
    main()