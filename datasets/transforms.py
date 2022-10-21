import torch
import torch.nn as nn
import torchvision.transforms as T

class SLResize():
    def __init__(self, size):
        self.imgT = T.Resize(size=size)

    def __call__(self, sample):
        sample['image'] = self.imgT(sample['image'])
        return sample


class SLCenterCrop():
    def __init__(self, size):
        self.imgT = T.CenterCrop(size=size)
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

    def __call__(self, sample):
        w, h = sample['image'].size
        sample['image'] = self.imgT(sample['image'])
        
        if 'landmarks' in sample:
            w_, h_ = self.size
            lm = sample['landmarks']
            lm_new = sample['landmarks'].copy()
            lm_new[:,0] = (lm[:,0] - 0.5) * w / w_ + 0.5
            lm_new[:,1] = (lm[:,1] - 0.5) * h / h_ + 0.5
            sample['landmarks'] = lm_new
        return sample


class SLToTensor():
    def __init__(self):
        self.imgT = T.ToTensor()

    def __call__(self, sample):
        sample['image'] = self.imgT(sample['image'])
        return sample

class SLNormalize():
    def __init__(self, mean, std):
        self.imgT = T.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        sample['image'] = self.imgT(sample['image'])
        return sample

class SLRandomResizedCrop():
    def __init__(self, size) -> None:
        self.imgT = T.RandomResizedCrop(size=size)

    def __call__(self, sample):
        sample['image'] = self.imgT(sample['image'])
        return sample
        
class SLRandomRotation():
    def __init__(self, degrees) -> None:
        self.imgT = T.RandomRotation(degrees=degrees)

    def __call__(self, sample):
        sample['image'] = self.imgT(sample['image'])
        return sample
