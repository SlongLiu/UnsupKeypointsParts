# Transformers
# - Lm Crop: Cropping image patches to given size (Receive PIL image)
# - Rescale: Resize image to square size by PIL ANTIALIAS operation (Receive PIL image, landmarks will be rescaled to [0~1])
# - ToTensor: Convert to image tensor (Receive PIL image)
# - Rotate: Rotate image patch by a random degree(90,180,270,360) (Receive ndarray or tensor)
# - Normalize: Normalize image matrix (Receive ndarray or tensor)

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

'''
Transformations for landmarks labelled datasets.

According to the training goal, the dataset will be converted
Dataset should be dict{'image': imgdata, 'label': labeldata}
'''


class BaseTransformer(object):
    """Base transformer for all transformers.

    Args:
        is_labeled(bool): The annotation status of given dataset.
    """

    def __init__(self, is_labeled=False):
        self.is_labeled = is_labeled

class LandmarkCrop():
    """crop n+1 lm on the Image

    Input:
        img: (PIL.Image)

    Returns:
        [type]: [description]
    """
    def __init__(self, heatmap_sigma, img_size, heatmap_size) -> None:
        self.additive_para = ['landmarks']
        self.heatmap_sigma = heatmap_sigma
        self.img_size = img_size 
        self.heatmap_size = heatmap_size

    def __call__(self, img, landmarks):
        lm_a = torch.rand(1, 2).numpy()
        lm_t = np.concatenate((landmarks, lm_a))
        bound_t = np.concatenate((lm_t - 3 * self.heatmap_sigma, lm_t + 3 * self.heatmap_sigma))
        bound_t = (bound_t * self.img_size / self.heatmap_size).astype(int)
        imglist = [img.crop(x) for x in bound_t]
        return imglist

class Normalize_batch:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor