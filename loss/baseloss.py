
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import gifdict

def l2_reconstruction_loss(x, x_, loss_mask=None):
    loss = (x - x_) ** 2
    if loss_mask is not None:
        loss = loss * loss_mask.to(loss.device)
    return torch.mean(loss)

def l1_reconstruction_loss(x, x_, loss_mask=None):
    loss = torch.abs(x - x_)
    if loss_mask is not None:
        loss = loss * loss_mask.to(loss.device)
    return torch.mean(loss)

def neg_log_dist_loss(gt_offset, pred_offset, e=1e-4):
    # delta = torch.abs(pred_offset - gt_offset)
    delta = torch.sqrt((pred_offset - gt_offset)**2 + e^2)
    loss = -torch.mean(torch.log((1 - delta).clamp(min=1e-10, max=1-1e-10)))
    return loss

def l2_dist_loss(gt_offset, pred_offset):
    loss = (pred_offset - gt_offset)**2
    return loss


class BaseLoss(nn.Module):
    def __init__(self, item_map=dict(x='x', y='y')):
        super(BaseLoss, self).__init__()
        self.item_map = item_map

    def run(self):
        raise NotImplementedError

    def forward(self, ddata):
        if isinstance(self.item_map, list):
            res = {}
            for idx, _item_map in enumerate(self.item_map):
                result_i = self.run(**{k:gifdict(ddata, v) for k,v in _item_map.items()})
                if not isinstance(result_i, dict):
                    result_i = {str(idx): result_i}
                else:
                    result_i = {str(idx)+'-'+k: v for k,v in result_i.items()}
                res.update(result_i)
            return res
        else:
            return self.run(**{k:gifdict(ddata, v) for k,v in self.item_map.items()})