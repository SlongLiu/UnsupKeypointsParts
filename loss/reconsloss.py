import torch
import torch.nn as nn
import numpy as np


from .baseloss import BaseLoss

def scale_points(x):
    return (x-0.5)*2

def rescale_points(x):
    return x/2+0.5

class EquivarianceLoss(BaseLoss):
    def __init__(self, item_map=dict(lm1='lm1', lm2='lm2', theta='theta'), **kw):
        super(EquivarianceLoss, self).__init__(item_map)
        self.criterion = nn.MSELoss()
        
    def run(self, lm1, lm2, theta):
        """loss = ||lm2*theta - lm1||

        Args:
            lm1 ([type]): B,N,2
            lm2 ([type]): B,N,2
            theta ([type]): B,2,3
        """
        B,N,_ = lm1.shape
        # print('theta.shape:', theta.shape)
        lm1_pred = torch.cat((scale_points(lm2), torch.ones(B,N,1).to(lm2.device)), dim=2).bmm(theta.permute(0,2,1))
        lm1_pred = rescale_points(lm1_pred)
        return self.criterion(lm1, lm1_pred)


class PointDistVisibLoss(BaseLoss):
    def __init__(self, loss_type='log', item_map=dict(x='x', y='y', v='v'), **kw):
        super(PointDistVisibLoss, self).__init__(item_map)
        self.loss_type = loss_type
        assert loss_type in ['log', 'l2']

    def run(self, x, y, v):
        """[summary]

        Args:
            x (B,N,2): [description]
            y (B,N,2): [description]
            v (B,N): [description]
        """
        # print('x.shape:', x.shape, 'y.shape:', y.shape)
        delta = x - y
        if self.loss_type == 'log':
            loss = -torch.mean(torch.log((1 - torch.abs(delta)).clamp(min=1e-10, max=1-1e-10)), dim=-1)
        else:
            loss = (delta**2).mean(-1)
        loss = (loss * v).mean()
        return loss

class SaliencyLoss(BaseLoss):
    def __init__(self, item_map=dict(saliency='saliency', mask_pred='mask_pred'), **kw):
        super(SaliencyLoss, self).__init__(item_map)
        
    def run(self, saliency, mask_pred):
        if saliency.dim() > 3:
            saliency = saliency.mean(1)
        return ((saliency-mask_pred)**2).mean()
