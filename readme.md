# SLIMM

This is my repo used for unsupervised part segmentation and keypoints discovery. 

It includes the implementation of our paper: 

[Unsupervised Part Segmentation through Disentangling Appearance and Shape.](https://arxiv.org/abs/2105.12405) CVPR 2021 

as well as a re-implementation of IMM:

[Unsupervised Learning of Object Landmarks through Conditional Image Generation](https://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/) (NeurIPS) 2018


## Run
```bash
# part segmentation (UnsupPartSeg)
python train_SCOPSP.py \
 -s path/to/log/dir \
 -c configs/SCOPSP/SCOPSP_CUB_BG_Norm_128x128_TPS_PERCEPNEW_ARC_10lm_hm32_newtc.py \
 -g 0 # gpu id(s)

# keypoints discovery (IMM, keypoints discovery)
python train_IMM.py \
 -s path/to/log/dir \
 -c configs/IMM/MyIMM_WILDCELEBA_128x128_TPS_PERCEP_10lm_h32_mask.py \
 -g 2,3 # gpu id(s)
```




