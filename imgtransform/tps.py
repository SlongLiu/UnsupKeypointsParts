# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import torch
import torch.nn.functional as F


def tps(theta, ctrl, grid):
    '''Evaluate the thin-plate-spline (TPS) surface at xy locations arranged in a grid.
    The TPS surface is a minimum bend interpolation surface defined by a set of control points.
    The function value for a x,y location is given by

        TPS(x,y) := theta[-3] + theta[-2]*x + theta[-1]*y + \sum_t=0,T theta[t] U(x,y,ctrl[t])

    This method computes the TPS value for multiple batches over multiple grid locations for 2
    surfaces in one go.

    Params
    ------
    theta: Nx(T+3)x2 tensor, or Nx(T+2)x2 tensor
        Batch size N, T+3 or T+2 (reduced form) model parameters for T control points in dx and dy.
    ctrl: NxTx2 tensor or Tx2 tensor
        T control points in normalized image coordinates [0..1]
    grid: NxHxWx3 tensor
        Grid locations to evaluate with homogeneous 1 in first coordinate.

    Returns
    -------
    z: NxHxWx2 tensor
        Function values at each grid location in dx and dy.
    '''

    N, H, W, _ = grid.size()

    if ctrl.dim() == 2:
        ctrl = ctrl.expand(N, *ctrl.size())

    T = ctrl.shape[1] # T

    diff = grid[..., 1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1) #N,H,W,T,2
    D = torch.sqrt((diff ** 2).sum(-1)) #N,H,W,T
    U = (D ** 2) * torch.log(D + 1e-6)  #N,H,W,T

    w, a = theta[:, :-3, :], theta[:, -3:, :] # N,T,2    N,3,2

    reduced = T + 2 == theta.shape[1]
    if reduced:
        w = torch.cat((-w.sum(dim=1, keepdim=True), w), dim=1)

        # U is NxHxWxT
    b = torch.bmm(U.view(N, -1, T), w).view(N, H, W, 2)
    # b is NxHxWx2

    # delete the rotation of the points.
    # z = torch.bmm(grid.view(N, -1, 3), a).view(N, H, W, 2) + b
    z = b

    return z


def tps_grid(theta, ctrl, size):
    '''Compute a thin-plate-spline grid from parameters for sampling.

    Params
    ------
    theta: Nx(T+3)x2 tensor
        Batch size N, T+3 model parameters for T control points in dx and dy.
    ctrl: NxTx2 tensor, or Tx2 tensor
        T control points in normalized image coordinates [0..1]
    size: tuple
        Output grid size as NxCxHxW. C unused. This defines the output image
        size when sampling.

    Returns
    -------
    grid : NxHxWx2 tensor
        Grid suitable for sampling in pytorch containing source image
        locations for each output pixel.
    '''
    N, _, H, W = size

    grid = theta.new(N, H, W, 3)
    grid[:, :, :, 0] = 1.
    grid[:, :, :, 1] = torch.linspace(0, 1, W)
    grid[:, :, :, 2] = torch.linspace(0, 1, H).unsqueeze(-1)

    z = tps(theta, ctrl, grid)
    return (grid[..., 1:] + z) * 2 - 1  # [-1,1] range required by F.sample_grid


def tps_sparse(theta, ctrl, xy):
    if xy.dim() == 2:
        xy = xy.expand(theta.shape[0], *xy.size())

    N, M = xy.shape[:2]
    grid = xy.new(N, M, 3)
    grid[..., 0] = 1.
    grid[..., 1:] = xy

    z = tps(theta, ctrl, grid.view(N, M, 1, 3))
    return xy + z.view(N, M, 2)


def uniform_grid(shape):
    '''Uniformly places control points aranged in grid accross normalized image coordinates.

    Params
    ------
    shape : tuple
        HxW defining the number of control points in height and width dimension
    Returns
    -------
    points: HxWx2 tensor
        Control points over [0,1] normalized image range.
    '''
    H, W = shape[:2]
    c = torch.zeros(H, W, 2)
    c[..., 0] = torch.linspace(0, 1, W)
    c[..., 1] = torch.linspace(0, 1, H).unsqueeze(-1)
    return c


def tps_sample_params(batch_size, num_control_points, var=0.05):
    theta = torch.randn(batch_size, num_control_points + 3, 2) * var
    cnt_points = torch.rand(batch_size, num_control_points, 2)
    return theta, cnt_points


def tps_transform(x, theta, cnt_points):
    device = x.device
    grid = tps_grid(theta, cnt_points, x.shape).type_as(x).to(device)
    return F.grid_sample(x, grid, padding_mode='zeros')


# class RandomTPSTransform(object):
#     def __init__(self, num_control=5, variance=0.05):
#         self.num_control = num_control
#         self.var = variance
#
#     def __call__(self, x, lossmask):
#         theta, cnt_points = tps_sample_params(x.size(0), self.num_control, self.var)
#         return tps_transform(x, lossmask, theta, cnt_points)

def rotate_affine_grid_multi(x, theta):
    theta = theta.to(x.device)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    transform = torch.zeros(x.size(0), 2, 3, dtype=x.dtype)
    transform[:, 0, 0] = cos_theta
    transform[:, 0, 1] = sin_theta
    transform[:, 1, 0] = - sin_theta
    transform[:, 1, 1] = cos_theta

    grid = F.affine_grid(transform, x.shape).to(x.device)
    return F.grid_sample(x, grid, padding_mode='zeros')


def rand_peturb_params(batch_items, tps_cntl_pts, tps_variance, max_rotate):
    theta_tps, cntl_pts = tps_sample_params(batch_items, tps_cntl_pts, tps_variance)
    theta_rotate = torch.rand(batch_items) * 2 - 1
    theta_rotate = theta_rotate * max_rotate
    return theta_tps, cntl_pts, theta_rotate


def peturb(x, tps_theta, cntl_pts, theta_rotate):
    x = tps_transform(x, tps_theta, cntl_pts)
    x = rotate_affine_grid_multi(x, theta_rotate)
    return x


def nop(*data):
    return data[0], data[1], None

class randTPSTransform():
    def __init__(self, num_control=5, tps_variance=0.05, rotate_variance=0.05):
        self.tps_cntl_pts = num_control
        self.tps_variance = tps_variance
        self.max_rotate = rotate_variance

    def __call__(self, x, return_mask=False):
        bsize = x.size(0)
        theta_tps, cntl_pts, theta_rotate = rand_peturb_params(bsize, self.tps_cntl_pts, self.tps_variance,
                                                               self.max_rotate) 
                                                    
        x1 = peturb(x, theta_tps, cntl_pts, theta_rotate)
        paras = {
            'tps_theta': theta_tps,
            'cntl_pts': cntl_pts,
            'theta_rotate': theta_rotate
        }

        # gen mask
        if return_mask:
            # image_mask = torch.ones(x.shape, dtype=x.dtype, device=x.device)
            # image_mask = peturb(image_mask, theta_tps, cntl_pts, theta_rotate)
            image_mask = (x1 > 1e-5).type(torch.FloatTensor)
            return x1, paras, image_mask

        return x1, paras
        

class TPS_Twice(object):
    def __init__(self, num_control=5, tps_variance=0.05, rotate_variance=0.05):
        """

        Args:
            num_control (int): Number of TPS control points
            variance (float): Variance of TPS transform coefficients
        """
        self.tps_cntl_pts = num_control
        self.tps_variance = tps_variance
        self.max_rotate = rotate_variance
        self.render = peturb

    def __call__(self, x):
        loss_mask = torch.ones(x.shape, dtype=x.dtype, device=x.device)
        bsize = x.size(0)
        theta_tps, cntl_pts, theta_rotate = rand_peturb_params(bsize, self.tps_cntl_pts, self.tps_variance,
                                                               self.max_rotate)
        x1 = peturb(x, theta_tps, cntl_pts, theta_rotate)
        mask1 = peturb(loss_mask, theta_tps, cntl_pts, theta_rotate)

        theta_tps, cntl_pts, theta_rotate = rand_peturb_params(bsize, self.tps_cntl_pts, self.tps_variance,
                                                               self.max_rotate)
        x2 = peturb(x, theta_tps, cntl_pts, theta_rotate)
        mask2 = peturb(loss_mask, theta_tps, cntl_pts, theta_rotate)
        return x1, mask1, x2, mask2


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from PIL import Image
    import numpy as np

    img = Image.open('/data/shilong/g32/code/IMM_pytorch/pics/gakki3_01.png').resize((128,128))
    x = torch.Tensor(np.array(img)/255).permute(2,0,1).unsqueeze(0)
    # x = torch.randn(1, 3, 128, 128)
    lossmask = torch.ones(1, 3, 128, 128)
    trans = TPS_Twice()
    x1, m1, x2, m2 = trans(x)
    import matplotlib.pyplot as plt
    # x1 = (x1 - x1.min()) / (x1.max() - x1.min())
    # x2 = (x2 - x2.min()) / (x2.max() - x2.min())
    print(x[0])

    fig, ax = plt.subplots(3, 2, figsize=(4, 6), dpi=80)
    ax[0,0].imshow(x1[0].permute(1, 2, 0))
    ax[0,1].imshow(m1[0].permute(1, 2, 0))
    ax[1,0].imshow(x2[0].permute(1, 2, 0))
    ax[1,1].imshow(m2[0].permute(1, 2, 0))
    ax[2,0].imshow((x[0]).permute(1, 2, 0))

    # ax[idx, i].set_title(SHORTNAME[_method])
    # ax.axis('off')
    # plt.imshow(m1[0].permute(1, 2, 0))
    # plt.savefig('1.png')
    # plt.show()
    # plt.imshow(m2[0].permute(1, 2, 0))
    plt.savefig('000.png')
    plt.show()
