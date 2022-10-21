
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import models, transforms
import torchvision
from torchvision.models import resnet

# from utils.transformers_sl import Normalize_batch

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

class discriminater(nn.Module):
    def __init__(self, modelname='resnet50', pretrained=True, usegpu=True):
        super().__init__()
        if modelname == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(512 * 4, 2)
        elif modelname == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.model.classifier[-1] = nn.Linear(4096, 2)
        else:
            raise NotImplementedError("model `%s` not implemented yet" % modelname)
        
        if not usegpu:
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        self.normalize = Normalize_batch(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            device=self.device
        )

        if modelname == 'MyD':
            self.lossfunc = nn.BCELoss()
            self.lossfuncname = 'BCELoss'
            self.labeltype = torch.float32
        else:
            self.lossfunc = nn.CrossEntropyLoss()
            self.lossfuncname = 'CrossEntropyLoss'
            self.labeltype = torch.long
        

    def get_loss(self, img, label_gt):
        img = self.normalize(img)
        label_pred = self.model(img)
        loss = self.lossfunc(label_pred, label_gt)
        return loss

    def forward(self, img, label):
        if label == 1:
            label_gt = torch.ones(img.size(0)).type(self.labeltype).to(img.device)
        else:
            label_gt = torch.zeros(img.size(0)).type(self.labeltype).to(img.device)

        return self.get_loss(img, label_gt)