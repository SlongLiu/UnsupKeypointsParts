import torch
import torch.nn as nn
import numpy as np

from loss.baseloss import BaseLoss

class PixelConnectLoss(BaseLoss):
    def __init__(self, item_map=dict(pred='pred'), **kw):
        super(PixelConnectLoss, self).__init__(item_map)
        self.padding = nn.ReflectionPad2d(padding=1)

    def run(self, pred):
        # pred: B,N,H,W
        B,N,H,W = pred.shape
        e_pred = self.padding(pred)
        # print(e_pred.shape)
        pixel_delta_list = []
        for delta_x, delta_y in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            tmp_pred = e_pred[:, :, 1+delta_y:(H+1)+delta_y, 1+delta_x:(W+1)+delta_x]
            # print('tmp_pred.shape:', tmp_pred.shape)
            pixel_delta = (tmp_pred - pred)**2
            pixel_delta_list.append(pixel_delta.unsqueeze(-1))
        pixel_delta_value = torch.min(torch.cat(pixel_delta_list), dim=-1)[0] # B,N,H,W
        loss = pixel_delta_value.sum((-1,-2)).mean()
        return loss

        

