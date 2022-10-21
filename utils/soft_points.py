# ==========================================================
# Author: Shilong Liu
# ==========================================================
import numpy as np
import torch
from torch._C import device

"""soft dual rep"""

def squared_diff(h, height):
    hs = torch.linspace(0, 1, height, device=h.device).type_as(h).expand(h.shape[0], h.shape[1], height)
    hm = h.expand(height, -1, -1).permute(1, 2, 0)
    hm = ((hs - hm) ** 2)
    return hm


def gaussian_like_function(kp, height, width, sigma=0.1, eps=1e-6):
    """from landmarks to gaussian like heatmap

    Args:
        kp ([type]): [description]
        height ([type]): [description]
        width ([type]): [description]
        sigma (float, optional): [description]. Defaults to 0.1.
        eps ([type], optional): [description]. Defaults to 1e-6.

    Returns:
        hm (Tensor): [description]
    """
    hm = squared_diff(kp[:, :, 0], height)
    wm = squared_diff(kp[:, :, 1], width)
    hm = hm.expand(width, -1, -1, -1).permute(1, 2, 3, 0)
    wm = wm.expand(height, -1, -1, -1).permute(1, 2, 0, 3)
    gm = - (hm + wm + eps).sqrt_() / (2 * sigma ** 2)
    gm = torch.exp(gm)
    return gm

def get_sigma(hm, mu):
    B,N,H,W = hm.shape
    assert (B,N,2) == mu.shape
    device = hm.device

    x_mean = mu[:,:,0].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) # BxNxHxW
    y_mean = mu[:,:,1].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) # BxNxHxW
    y_ind = torch.linspace(0, 1, H).unsqueeze(-1).repeat(B, N, 1, W).to(device)
    x_ind = torch.linspace(0, 1, W).unsqueeze(0).repeat(B, N, H, 1).to(device)

    y_sd = torch.sqrt(((y_ind - y_mean)**2 * hm).mean((-1,-2))) # B,N
    x_sd = torch.sqrt(((x_ind - x_mean)**2 * hm).mean((-1,-2)))
    return torch.cat((x_sd.unsqueeze(-1), y_sd.unsqueeze(-1)), dim=-1)


def get_gaussian_mean(x, axis, other_axis, softmax=True):
    """

    Args:
        x (float): Input images(BxCxHxW)
        axis (int): The index for weighted mean
        other_axis (int): The other index

    Returns: weighted index for axis, BxC

    """
    mat2line = torch.sum(x, axis=other_axis)
    # mat2line = mat2line / mat2line.mean() * 10
    if softmax:
        u = torch.softmax(mat2line, axis=2)
    else:
        u = mat2line / (mat2line.sum(2, keepdim=True) + 1e-6)
    size = x.shape[axis]
    ind = torch.linspace(0, 1, size).to(x.device)
    batch = x.shape[0]
    channel = x.shape[1]
    index = ind.repeat([batch, channel, 1])
    mean_position = torch.sum(index * u, dim=2)
    return mean_position

# def get_gaussian_map_from_points(points, height, weight, std, device, eps=1e-6):
#     """get_gaussian_map_from_points
#         B,N,2 float(0, 1) float(0, 1) -> B,C,H,W

#     Args:
#         points (torch.Tensor B,N,2): [description]
#         height (int): [description]
#         weight (int): [description]
#         std (float): [description]
#         device (device): [description]
#         eps (float, optional): [description]. Defaults to 1e-6.

#     Returns:
#         [type]: [description]
#     """
#     H, W = height, weight
#     B, N, _ = points.size()
#     x_mean = points[:,:,0].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) # BxNxHxW
#     y_mean = points[:,:,1].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) # BxNxHxW
#     y_ind = torch.linspace(0, 1, H).unsqueeze(-1).repeat(B, N, 1, W).to(device)
#     x_ind = torch.linspace(0, 1, W).unsqueeze(0).repeat(B, N, H, 1).to(device)
#     dist = (x_ind - x_mean) ** 2 + (y_ind - y_mean) ** 2
#     res = torch.exp(-(dist + 1e-6).sqrt_() / (2 * std ** 2))
#     return res

def get_expected_points_from_map(hm, softmax=True):
    """get_gaussian_map_from_points
        B,C,H,W -> B,N,2 float(0, 1) float(0, 1)
        softargmax function

    Args:
        hm (float): Input images(BxCxHxW)

    Returns: 
        weighted index for axis, BxCx2. float between 0 and 1.

    """
    # hm = 10*hm
    B,C,H,W = hm.shape
    y_mean = get_gaussian_mean(hm, 2, 3, softmax=softmax) # B,C
    x_mean = get_gaussian_mean(hm, 3, 2, softmax=softmax) # B,C
    # return torch.cat((x_mean.unsqueeze(-1), y_mean.unsqueeze(-1)), 2)
    return torch.stack([x_mean, y_mean], dim=2)

"""soft dual rep end"""

def get_gaussian_map_from_points(points, height, weight, std, device=None, eps=1e-5, mode='point'):
    """[summary]

    Args:
        points (B,N,2): [description]
        height ([type]): [description]
        weight ([type]): [description]
        std ([type]): [description]
        device ([type]): [description]
        eps ([type], optional): [description]. Defaults to 1e-6.
        mode (str, optional): [description]. Defaults to 'ankush'.
    """
    B, N, _ = points.shape
    H, W = height, weight
    if device is None:
        device = points.device
    mu_x, mu_y = points[:, :, 0:1], points[:, :, 1:2] #B,N,1
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)

    if mode in ['rot', 'flat', 'point']:
        mu_x = mu_x.unsqueeze_(-1) 
        mu_y = mu_y.unsqueeze_(-1) # B,N,1,1

        y = y.reshape(1, 1, H, 1)
        x = x.reshape(1, 1, 1, W)
        dist = ((y - mu_y)**2 + (x - mu_x)**2) * std**2 # B,N,H,W

        if mode == 'rot':
            g_yx = torch.exp(-(4*dist))
        elif mode == 'flat':
            g_yx = torch.exp(-torch.pow(dist + eps, 0.25))
        else:
            g_yx = torch.exp(-(dist + 1e-6).sqrt_() / (2 * std ** 2))

    elif mode == 'ankush':
        y = y.reshape(1, 1, H)
        x = x.reshape(1, 1, W)

        g_y = torch.exp(-torch.sqrt(torch.abs((y - mu_y) * std) + eps))
        g_x = torch.exp(-torch.sqrt(torch.abs((x - mu_x) * std) + eps))
        
        g_y = g_y.unsqueeze(3) # B,N,H,1
        g_x = g_x.unsqueeze(2) # B,N,1,W
        g_yx = torch.matmul(g_y, g_x)
    
    else:
        raise ValueError("Unknown mode: %s" % mode)

    return g_yx




