from typing import Type
from numpy.lib.arraysetops import isin
import torch.nn.functional as F
import torch
import numpy as np
# import math


def spatial_transform(img, theta):
    """[summary]

    Args:
        img (torch.Tensor): Bx3xHxW
        theta (torch.Tensor): Bx2x3

    Returns:
        new_img (torch.Tensor): Bx3xHxW
    """
    device = img.device
    grid = F.affine_grid(theta, img.size()).to(device)
    output = F.grid_sample(img, grid)
    return output

def convert_tensor(x):
    if not isinstance(x, torch.Tensor):
        if not isinstance(x, list):
            x = [x]
        return torch.Tensor(x)
    return x

def ST_transform(img, offset_x, offset_y, angle, scale_theta):
    # convert to torch.Tensor
    offset_x = convert_tensor(offset_x)
    offset_y = convert_tensor(offset_y)
    angle = convert_tensor(angle)
    scale_theta = convert_tensor(scale_theta)

    b = img.size(0)
    theta_mat = torch.zeros(b, 2, 3)
    # rotate
    theta_mat[:, 0, 0] = torch.cos(angle)
    theta_mat[:, 0, 1] = torch.sin(-angle)
    theta_mat[:, 1, 0] = torch.sin(angle)
    theta_mat[:, 1, 1] = torch.cos(angle)
    # scale
    theta_mat = theta_mat * scale_theta.unsqueeze(-1).unsqueeze(-1)
    # offset
    theta_mat[:, 0, 2] = offset_x
    theta_mat[:, 1, 2] = offset_y

    return spatial_transform(img, theta_mat)

class randST():
    def __init__(self, offset_x_range=(-0.5, 0.5), offset_y_range=(-0.5, 0.5), rotate_range=None, scale_range=None):
        """init

        Arguments:
            offset_x_range {tuple(float) or None} -- offset y range. (-1, 1)
            offset_y_range {tuple(float) or None} -- offset y range. (-1, 1)
            rotate_range {tuple(float) or None} -- max rotate in rad. (-1, 1) crt (-\pi, \pi)
            scale_range {tuple(float) or None} -- scale range. (0 to \infty)
        """
        self.offset_x_range = offset_x_range
        self.offset_y_range = offset_y_range
        self.rotate_range = rotate_range
        self.scale_range = scale_range

    def __call__(self, img):
        """take the rigid body transformation of the img with random parameters.

        Arguments:
            img {Tensor(NXCXHXW)} -- a batch of imgs

        Returns:
            img {Tensor(NXCXHXW)} -- imgs after transformation
        """
        b = img.size(0)
        if self.offset_x_range is None:
            offset_x = torch.zeros(b)
        else:
            offset_x = torch.rand(b) * (self.offset_x_range[1] - self.offset_x_range[0]) + self.offset_x_range[0]

        if self.offset_y_range is None:
            offset_y = torch.zeros(b)
        else:
            offset_y = torch.rand(b) * (self.offset_y_range[1] - self.offset_y_range[0]) + self.offset_y_range[0]

        if self.rotate_range is None:
            angle = torch.zeros(b)
        else:
            angle = (torch.rand(b) * (self.rotate_range[1] - self.rotate_range[0]) + self.rotate_range[0])
            angle = angle * np.pi

        if self.scale_range is None:
            scale_theta = torch.ones(b)
        else:
            scale_theta = torch.rand(b) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]

        img_new = ST_transform(img, offset_x, offset_y, angle, scale_theta)

        # theta_mat = torch.zeros(b, 2, 3)
        # # rotate
        # theta_mat[:, 0, 0] = torch.cos(angle)
        # theta_mat[:, 0, 1] = torch.sin(-angle)
        # theta_mat[:, 1, 0] = torch.sin(angle)
        # theta_mat[:, 1, 1] = torch.cos(angle)
        # # scale
        # theta_mat = theta_mat * scale_theta.unsqueeze(-1).unsqueeze(-1)
        # # offset
        # theta_mat[:, 0, 2] = offset_x
        # theta_mat[:, 1, 2] = offset_y

        return img_new, offset_x, offset_y, angle, scale_theta
    

class ST_twice():
    def __init__(self, **kw):
        self.rst = randST(**kw)
        self.kw = kw

    def __call__(self, x):
        loss_mask = torch.ones(x.shape, dtype=x.dtype, device=x.device)
        x1, offset_x1, offset_y1, angle1, scale_theta1 = self.rst(x)
        mask1 = ST_transform(loss_mask, offset_x1, offset_y1, angle1, scale_theta1)

        x2, offset_x2, offset_y2, angle2, scale_theta2 = self.rst(x)
        mask2 = ST_transform(loss_mask, offset_x2, offset_y2, angle2, scale_theta2)

        return x1, mask1, x2, mask2

def get_random_para(N, item_range=None, log_scale=False):
    if not log_scale:
        if item_range is None:
            return torch.zeros(N)
        if isinstance(item_range, (int, float)):
            return torch.ones(N) * item_range
        if isinstance(item_range, tuple):
            return torch.rand(N) * (item_range[1] - item_range[0]) + item_range[0] 
        raise TypeError(f"Unknown type {type(item_range)}")
    else:
        if item_range is None:
            return torch.ones(N)
        if isinstance(item_range, (int, float)):
            return torch.ones(N) * item_range
        if isinstance(item_range, tuple):
            res = torch.rand(N) * (np.log(item_range[1]) - np.log(item_range[0])) + np.log(item_range[0]) 
            res = torch.exp(res)
            return res
        raise TypeError(f"Unknown type {type(item_range)}")



class randSpatialTransform():
    def __init__(self, mode='parameters', paras={}, align_corners=False):
        """init

        Args:
            mode (str, optional): parameters or matrix. Defaults to 'parameters'.
            paras: keys if mode=='parameters':  offset_x_range, offset_y_range: (-1, 1)
                                                rotate_range: (-1, 1)
                                                scale_range: (0.5, 2)
                                                shear_x_range, shear_y_range: (-1, 1)
                                                all of which are tuple(float, float)
                        if matrix mode: matrix_max: Tensor(2X3), matrix_min: Tensor(2X3)
        """
        assert mode == 'parameters' or mode == 'matrix'
        self.paras = paras
        self.mode = mode
        self.align_corners = align_corners
        self.render = self.spatial_transform

    def gen_parameters(self, N):
        # random gen parameters
        offset_x = get_random_para(N, self.paras.get('offset_x_range'))
        offset_y = get_random_para(N, self.paras.get('offset_y_range'))
        angle = get_random_para(N, self.paras.get('rotate_range')) * np.pi
        scale_theta = get_random_para(N, self.paras.get('scale_range'), True)
        shear_x = get_random_para(N, self.paras.get('shear_x_range'))
        shear_y = get_random_para(N, self.paras.get('shear_y_range'))

        return offset_x, offset_y, angle, scale_theta, shear_x, shear_y


    def gen_matrix(self, N):
        # random gen matrix
        matrix_max = self.paras.get('matrix_max')
        matrix_min = self.paras.get('matrix_min')
        I = torch.zeros(N, 2, 3)
        I[:, 0, 0] = 1
        I[:, 1, 1] = 1
        if matrix_max is None and matrix_min is None:
            return I
        if matrix_max is None:
            matrix_max == torch.max(I, matrix_min)

        if matrix_min is None:
            matrix_min == torch.min(I, matrix_max)

        matrix = torch.zeors(N, 2, 3)
        for i in range(2):
            for j in range(3):
                matrix[:, i, j] = torch.rand(N) * (matrix_max[i,j] - matrix_min[i,j]) + matrix_min[0]
        return matrix                
    

    def paras2matrix(self, b, offset_x, offset_y, angle, scale_theta, shear_x, shear_y):
        """transform from parameters to matrix

        Args:
            b (int): batch size
            offset_x ([type]): [description]
            offset_y ([type]): [description]
            angle ([type]): [description]
            scale_theta ([type]): [description]
            shear_x ([type]): [description]
            shear_y ([type]): [description]

        Returns:
            matrix: Nx2x3
        """
        theta_mat = torch.zeros(b, 2, 3)
        # rotate
        theta_mat[:, 0, 0] = torch.cos(angle)
        theta_mat[:, 0, 1] = torch.sin(-angle)
        theta_mat[:, 1, 0] = torch.sin(angle)
        theta_mat[:, 1, 1] = torch.cos(angle)
        # scale
        theta_mat = theta_mat * scale_theta.unsqueeze(-1).unsqueeze(-1)
        
        # shear_x
        mat_shear_x = torch.zeros(b, 3, 3)
        mat_shear_x[:, 0, 0] = 1
        mat_shear_x[:, 1, 1] = 1
        mat_shear_x[:, 0, 1] = shear_x
        # shear_y
        mat_shear_y = torch.zeros(b, 3, 3)
        mat_shear_y[:, 0, 0] = 1
        mat_shear_y[:, 1, 1] = 1
        mat_shear_y[:, 1, 0] = shear_y
        # bmm
        theta_mat = theta_mat.bmm(mat_shear_x).bmm(mat_shear_y)
        
        # offset
        theta_mat[:, 0, 2] = offset_x
        theta_mat[:, 1, 2] = offset_y

        return theta_mat
        
    def spatial_transform(self, img, theta):
        device = img.device
        grid = F.affine_grid(theta, img.size(), align_corners=self.align_corners).to(device)
        output = F.grid_sample(img, grid, align_corners=self.align_corners)
        return output

    def __call__(self, img, return_mask=False):
        """spatial transoform a batch of imgs.

        Arguments:
            img {Tensor(NXCXHXW)} -- a batch of imgs

        Returns:
            img {Tensor(NXCXHXW)} -- imgs after transformation
            matrix {Tensor(Nx2x3)} -- transform matrix
        """
        N, C, H, W = img.size()
        if self.mode == 'parameters':
            # random gen parameters
            offset_x, offset_y, angle, scale_theta, shear_x, shear_y = self.gen_parameters(N)
            # transform parameters to matrix
            matrix = self.paras2matrix(N, offset_x, offset_y, angle, scale_theta, shear_x, shear_y)
        else:
            matrix = self.gen_matrix(N)
        
        img_new = self.spatial_transform(img, matrix)

        # paras = {
        #     'theta': matrix
        # }
        paras = matrix.squeeze()

        # gen mask
        if return_mask:
            image_mask = torch.ones(img.shape, dtype=img.dtype, device=img.device)
            image_mask = self.spatial_transform(image_mask, matrix)
            return img_new, paras, image_mask

        return img_new, paras




