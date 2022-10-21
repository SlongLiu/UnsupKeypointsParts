

import torch
import numpy as np

from loss.baseloss import BaseLoss
from utils.vis_utils import get_coordinate_tensors, batch_get_centers
from utils.soft_points import get_gaussian_mean


def get_variance(part_map, x_c, y_c):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h,w)

    v_x_map = (x_map - x_c) * (x_map - x_c)
    v_y_map = (y_map - y_c) * (y_map - y_c)

    v_x = (part_map * v_x_map).sum()
    v_y = (part_map * v_y_map).sum()
    return v_x, v_y

def concentration_loss(pred):
    # pred_softmax = softmax(pred)[:,1:,:,:]
    B,C,H,W = pred.shape

    loss = 0
    epsilon = 1e-3
    centers_all = batch_get_centers(pred)
    for b in range(B):
        centers = centers_all[b]
        for c in range(C):
            # normalize part map as spatial pdf
            part_map = pred[b,c,:,:] + epsilon # prevent gradient explosion
            k = part_map.sum()
            part_map_pdf = part_map/k
            x_c, y_c = centers[c]
            v_x, v_y = get_variance(part_map_pdf, x_c, y_c)
            loss_per_part = (v_x + v_y)
            loss = loss_per_part + loss
    loss = loss/B
    return loss


def concentration_loss_new(pred, sqrt=False):
    # pred = pred[:,1:,...]
    B,C,H,W = pred.shape
    device = pred.device
    z_k = pred.sum((-1,-2), keepdim=True) + 1e-6
    pred = pred / torch.min(z_k, torch.ones(B,C,H,W).to(device))

    mu_y = get_gaussian_mean(pred, 2, 3, softmax=False).unsqueeze(-1).unsqueeze(-1) # B,C,1,1
    mu_x = get_gaussian_mean(pred, 3, 2, softmax=False).unsqueeze(-1).unsqueeze(-1) # B,C,1,1
    y = torch.linspace(0, 1, H, device=device) 
    x = torch.linspace(0, 1, W, device=device)
    y = y.reshape(1, 1, H, 1)
    x = x.reshape(1, 1, 1, W)
    dist = ((y - mu_y)**2 + (x - mu_x)**2) # B,C,H,W
    if sqrt:
        dist = dist ** 0.5

    loss = dist * pred 
    return loss.sum() / B

def concentration_loss_l1(pred, sqrt=False):
    # pred = pred[:,1:,...]
    B,C,H,W = pred.shape
    device = pred.device
    z_k = pred.sum((-1,-2), keepdim=True) + 1e-6
    pred = pred / torch.min(z_k, torch.ones(B,C,H,W).to(device))

    mu_y = get_gaussian_mean(pred, 2, 3, softmax=False).unsqueeze(-1).unsqueeze(-1) # B,C,1,1
    mu_x = get_gaussian_mean(pred, 3, 2, softmax=False).unsqueeze(-1).unsqueeze(-1) # B,C,1,1
    y = torch.linspace(0, 1, H, device=device) 
    x = torch.linspace(0, 1, W, device=device)
    y = y.reshape(1, 1, H, 1)
    x = x.reshape(1, 1, 1, W)
    dist = (torch.sqrt((y - mu_y)**2) + torch.sqrt((x - mu_x)**2)) # B,C,H,W
    if sqrt:
        dist = dist ** 0.5

    loss = dist * pred 
    return loss.sum() / B

def concentration_boundary_loss(pred, sqrt=False):
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    
    B,C,H,W = pred.shape
    device = pred.device
    z_k = pred.sum((-1,-2), keepdim=True) + 1e-6
    pred = pred / torch.min(z_k, torch.ones(B,C,H,W).to(device))

    y = torch.linspace(0, 1, H, device=device) 
    x = torch.linspace(0, 1, W, device=device)
    y = y.reshape(1, 1, H, 1)
    x = x.reshape(1, 1, 1, W)

    # mu_y = get_gaussian_mean(1-pred, 2, 3, softmax=False).unsqueeze(-1).unsqueeze(-1)
    # mu_x = get_gaussian_mean(1-pred, 3, 2, softmax=False).unsqueeze(-1).unsqueeze(-1) # B,C,1,1
    # dist = (())

    zeros = torch.zeros_like(pred, device=device) - 0.01
    ones = torch.ones_like(pred, device=device) + 0.01
    dist = torch.min(torch.min((x-zeros)**2, (x-ones)**2), torch.min((y-zeros)**2, (y-ones)**2))*2
    if sqrt:
        dist = dist ** 0.5

    # cc = torch.ones_like(pred, device=device) / 2
    # dist = (1 - torch.abs(x - cc))**2 + (1 - torch.abs(y - cc)) ** 2

    loss = dist * pred
    return loss.sum() / B


class ConcentrationLoss(BaseLoss):
    def __init__(self, item_map=dict(pred='pred'), bg=False, sqrt=False, **kw):
        super(ConcentrationLoss, self).__init__(item_map)
        self.bg = bg
        self.sqrt = sqrt

    def run(self, pred):
        if self.bg:
            pred = pred[:,1:,...]
        # return concentration_loss(pred)
        return concentration_loss_new(pred, self.sqrt)

class ConcentBGLoss(BaseLoss):
    def __init__(self, item_map=dict(pred='pred'), sqrt=False, **kw) -> None:
        super(ConcentBGLoss, self).__init__(item_map)
        self.sqrt = sqrt
        
    def run(self, pred):
        pred = pred[:,0,...].clone().unsqueeze(1)
        return concentration_boundary_loss(pred, self.sqrt)
        
class DeltaLoss(BaseLoss):
    def __init__(self, item_map=dict(pred='pred'), **kw):
        super().__init__(item_map)
        

    def run(self, pred):
        # pred: BCHW
        B,C,H,W = pred.shape
        delta_tuples = ((0,1), (1,0))
        epsilon = 1e-4

        pred_com = pred[:, :, 1:H-1, 1:W-1]
        loss = 0
        for dx, dy in delta_tuples:
            pred_i = pred[:, :, 1+dx:dx+H-1, 1+dy:dy+W-1]
            loss += ((pred_com - pred_i)**2).sum()/(B*C)
        
        return loss
