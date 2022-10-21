import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .baseloss import BaseLoss
from loss.baseloss import l1_reconstruction_loss, l2_reconstruction_loss
from utils.utils import AverageMeter, gifdict

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (x, h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.conv1_2 = nn.Sequential()
        self.conv2_2 = nn.Sequential()
        self.conv3_2 = nn.Sequential()
        self.conv4_2 = nn.Sequential()
        self.conv5_2 = nn.Sequential()

        for x in range(4):
            self.conv1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.conv2_2.add_module(str(x), features[x])
        for x in range(9, 14):
            self.conv3_2.add_module(str(x), features[x])
        for x in range(14, 23):
            self.conv4_2.add_module(str(x), features[x])
        for x in range(23, 32):
            self.conv5_2.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x: torch.Size([10, 3, 128, 128])
        feat1_2: torch.Size([10, 64, 128, 128])
        feat2_2: torch.Size([10, 128, 64, 64])
        feat3_2: torch.Size([10, 256, 32, 32])
        feat4_2: torch.Size([10, 512, 16, 16])
        feat5_2: torch.Size([10, 512, 8, 8])

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        h = self.conv1_2(x)
        feat1_2 = h
        h = self.conv2_2(h)
        feat2_2 = h
        h = self.conv3_2(h)
        feat3_2 = h
        h = self.conv4_2(h)
        feat4_2 = h
        h = self.conv5_2(h)
        feat5_2 = h
        out = (x, feat1_2, feat2_2, feat3_2, feat4_2, feat5_2)
        return out


def normlize(x):
    device = x.device
    _mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    _std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
    x = (x.permute(0, 2, 3, 1)-_mean)/_std
    return x.permute(0, 3, 1, 2) 


class PerceptualLoss(BaseLoss):
    def __init__(self, loss_type='L2', 
            model_type='Vgg19', 
            content_layer=None,
            item_map=dict(x='x', y='y', mask='mask'),
            layer_weight=None,
            refresh_iterval = 20,
            **kw):
        super(PerceptualLoss, self).__init__(item_map)
        self.loss_type = loss_type
        self.content_layer = content_layer
        if content_layer is None:
            content_layer = list(range(6))
        self.layer_weight = layer_weight
        if layer_weight is None:
            self.layer_weight = [1.0] * len(content_layer)
        # self.item_map = item_map

        # child
        assert model_type == 'Vgg19', ("only support Vgg19")
        self.vgg = getattr(sys.modules[__name__], model_type)()
        # self.vgg = Vgg19()
        if loss_type == 'L2':
            self.lossfunc = l2_reconstruction_loss
        elif loss_type == 'L1':
            self.lossfunc = l1_reconstruction_loss
        else:
            raise ValueError('Unsupported loss type %s' % loss_type)

        # mask shape
        self.mask_scale = {
            '0': 1,
            '1': 1,
            '2': 2,
            '3': 4,
            '4': 8,
            '5': 16
        }

        # moving average
        self.mov_a = [AverageMeter() for i in range(6)]
        self.refresh_iterval = refresh_iterval
        

    def run(self, x, y, mask=None):
        if mask is not None and mask.size(1) != 1:
            mask = mask.mean(1, keepdim=True)
        # mask = (mask-0.5)*2

        # xf = self.vgg(normlize(x * mask.to(x.device)))
        # yf = self.vgg(normlize(y * mask.to(y.device)))
        xf = self.vgg(normlize(x))
        yf = self.vgg(normlize(y))

        L_percep = {}
        for i in range(6):
            if mask is not None:
                if self.mask_scale[str(i)] > 1:
                    mask = F.avg_pool2d(mask, 2)

            loss_i = self.lossfunc(xf[i], yf[i], mask) * self.layer_weight[i]
            
            if i in self.content_layer:
                L_percep.update({str(i): loss_i})

            self.mov_a[i].update(loss_i.mean().item())      

            # refresh 
        if self.refresh_iterval != -1:
            for i in range(6):
                if (self.mov_a[i].cnt + 1) % self.refresh_iterval == 0:
                    # print([j._avg for j in self.mov_a])
                    self.layer_weight[i] = self.layer_weight[i] / (self.mov_a[i]._avg + 1e-6)
                    self.mov_a[i].reset()
    
        return L_percep


class PerceptualNewLoss(BaseLoss):
    def __init__(self, loss_type='L2', 
            model_type='Vgg19', 
            content_layer=None,
            item_map=dict(x='x', y='y', mask='mask'),
            layer_weight=None,
            refresh_iterval = 20,
            return_single_loss = True,
            **kw):
        super(PerceptualNewLoss, self).__init__(item_map)
        self.loss_type = loss_type
        self.content_layer = content_layer
        self.return_single_loss = return_single_loss
        if content_layer is None:
            content_layer = list(range(6))
        self.layer_weight = layer_weight
        if layer_weight is None:
            self.layer_weight = [1.0] * len(content_layer)
        # self.item_map = item_map

        # child
        assert model_type == 'Vgg19', ("only support Vgg19")
        self.vgg = getattr(sys.modules[__name__], model_type)()
        # self.vgg = Vgg19()
        if loss_type == 'L2':
            self.lossfunc = l2_reconstruction_loss
        elif loss_type == 'L1':
            self.lossfunc = l1_reconstruction_loss
        else:
            raise ValueError('Unsupported loss type %s' % loss_type)

        # mask shape
        self.mask_scale = {
            '0': 1,
            '1': 1,
            '2': 2,
            '3': 4,
            '4': 8,
            '5': 16
        }

        # moving average
        self.mov_a = [AverageMeter() for i in range(6)]
        self.refresh_iterval = refresh_iterval
        

    def run(self, x, y, mask=None):
        if mask is not None and mask.size(1) != 1:
            mask = mask.mean(1, keepdim=True)
        # mask = (mask-0.5)*2

        # xf = self.vgg(normlize(x * mask.to(x.device)))
        # yf = self.vgg(normlize(y * mask.to(y.device)))
        xf = self.vgg(normlize(x))
        yf = self.vgg(normlize(y))

        L_percep = {}
        loss_list = []
        loss_sum = 0
        for i in range(6):
            if mask is not None:
                if self.mask_scale[str(i)] > 1:
                    mask = F.avg_pool2d(mask, 2)
            loss_i_r = self.lossfunc(xf[i], yf[i], mask)
            loss_list.append(loss_i_r)
            loss_sum = loss_sum + loss_i_r
            self.mov_a[i].update(loss_i_r.mean().item())

            # if i in self.content_layer:
            #     L_percep.update({str(i): loss_i_r * self.layer_weight[i]})


        for i in range(6):
            loss_i = loss_list[i] * self.layer_weight[i]
            if i in self.content_layer:
                L_percep.update({str(i): loss_i * loss_sum / 6})
            # self.mov_a[i].update(loss_i.mean().item())      

            # refresh 
        if self.refresh_iterval != -1:
            for i in range(6):
                if (self.mov_a[i].cnt + 1) % self.refresh_iterval == 0:
                    # print([j._avg for j in self.mov_a])
                    self.layer_weight[i] = 1 / (self.mov_a[i]._avg + 1e-6)
                    self.mov_a[i].reset()
    
        if self.return_single_loss:
            loss = 0
            for k,v in L_percep.items():
                loss += v
            return loss

        return L_percep
        

class PointPenaltyLoss(BaseLoss):
    def __init__(self, item_map=dict(hm='hm', mask='mask'), **kw):
        super(PointPenaltyLoss, self).__init__(item_map)
    
    def run(self, hm, mask):
        mask = F.interpolate(mask, size=hm.shape[2:]).to(hm.device)
        loss = ((1-mask)*hm).mean()
        return loss



# class PerceptualFilter():
#     def __init__(self) -> None:
#         self.cnt = 0
    
#     def filter(lossdict):
