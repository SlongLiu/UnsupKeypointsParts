# ==========================================================
# Author: Shilong Liu
# ==========================================================
import numpy as np
import torch

""" absolute dual rep """

def get_points_from_heatmaps_batch(hm):
    """ get_points_from_heatmaps
        B,C,H,W -> B,N,2 int(0, H-1) int(0, W-1)

    Args:
        hm (torch.Tensor or np.ndarry): B,C,H,W

    Return:
        points (torch.Tensor or np.ndarry): B,C,2. \
            dtype: int. range: (0,H-1) (0,W-1)
    """
    if isinstance(hm, torch.Tensor):
        B, C, H, W = hm.size()
        device = hm.device

        hm_reshape = hm.reshape((B, C, -1))                                     #b,c,HxW
        idx = torch.argmax(hm_reshape, -1).unsqueeze(-1).repeat((1, 1, 2))      #b,c,2

        idx[:, :, 0] = idx[:, :, 0] % W
        idx[:, :, 1] = idx[:, :, 1] // W
    elif isinstance(hm, np.ndarray):
        B, C, H, W = hm.shape

        hm_reshape = hm.reshape((B, C, -1))                 #b,c,HxW
        idx = np.argmax(hm_reshape, -1).reshape((B, C, 1))      #b,c,1
        idx = np.tile(idx, (1, 1, 2))                           #b,c,2

        idx[:, :, 0] = idx[:, :, 0] % W
        idx[:, :, 1] = idx[:, :, 1] // W
    else:
        raise NotImplementedError("type %s is not supported." % str(type(hm)))

    return idx



def get_heatmaps_from_points(points, hm_size, sigma):
    """ get_heatmaps_from_points

        only np.ndarray is supported temporary.

    Args:
        points (torch.Tensor or np.ndarray): shape: C,2.
        hm_size (torch.Tensor or np.ndarray): shape: 2. Height x Width
        sigma (int): sigma of hm. the output will be in the 3 sigma range.
    """
    if isinstance(points, np.ndarray):
        C, _ = points.shape
        H, W = hm_size
        
        target = np.zeros((C, H, W))
        tmp_size = sigma * 3 

        for idx, point in enumerate(points):
            mu_x, mu_y = point

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= W or ul[1] >= H  or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], W) - ul[0] 
                # max(0, -ul[0]) = max(0, ul[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], W)
            img_y = max(0, ul[1]), min(br[1], H)

            # v = target_weight[joint_id]
            # rendering
            target[idx][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            
        return target
    else:
        raise NotImplementedError("Type `%s` is not supported yet." % str(type(points)))



def get_heatmaps_from_points_batch(points_batch, hm_size, sigma):
    """ get_heatmaps_from_points batch
        B,N,2 int(0, H-1) int(0, W-1) -> B,C,H,W

        only np.ndarray is supported temporary.

    Args:
        points (torch.Tensor or np.ndarray): shape: B,C,2.
        hm_size (torch.Tensor or np.ndarray): shape: 2. Height x Width
        sigma (int): sigma of hm. the output will be in the 3 sigma range.
    """      
    if isinstance(points_batch, np.ndarray):
        B, C, _ = points_batch.shape
        H, W = hm_size
        
        target = np.zeros((B, C, H, W))
        for idx, points in enumerate(points_batch):
            target[idx] = get_heatmaps_from_points(points, hm_size, sigma)
        return target
    else:
        raise NotImplementedError("Type `%s` is not supported yet." % str(type(points_batch)))

""" absolute dual rep end """



