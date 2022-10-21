
from utils.soft_points import get_gaussian_map_from_points
from network.immmodel import HeatMap, MyContentEncoder, MyGenerator, MyPoseEncoder
import os.path as osp
import sys
import numpy as np
# sys.path.append(osp.dirname(sys.path[0]))

import torch
import torch.nn as nn
import torchvision

import network.net_blocks as nb
from geoutils.geom_utils import orthographic_proj, orthographic_proj_withz

#------------- Modules ------------#
#----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x) # B,512,4,4
        return x

class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=n_blocks)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        # out: B, nz_feat
        resnet_feat = self.resnet_conv.forward(img)

        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv1)

        return feat



class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, n_kp):
        super(ShapePredictor, self).__init__()
        # self.pred_layer = nb.fc(True, nz_feat, n_kp)
        self.pred_layer = nn.Linear(nz_feat, n_kp * 3)

        # Initialize pred_layer weights to be small so initial def aren't so big
        self.pred_layer.weight.data.normal_(0, 0.0001)

    def forward(self, feat):
        # pdb.set_trace()
        delta_v = self.pred_layer.forward(feat)
        # Make it B x n_kp x 3
        delta_v = delta_v.view(delta_v.size(0), -1, 3)
        # print('shape: ( Mean = {}, Var = {} )'.format(delta_v.mean().data[0], delta_v.var().data[0]))
        return delta_v


class QuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot=4, classify_rot=False):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)
        self.classify_rot = classify_rot

    def forward(self, feat):
        quat = self.pred_layer.forward(feat)
        if self.classify_rot:
            quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)
        return quat


class ScalePredictor(nn.Module):
    def __init__(self, nz):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)

    def forward(self, feat):
        scale = self.pred_layer.forward(feat) + 1  #biasing the scale to 1
        scale = torch.nn.functional.relu(scale) + 1e-12
        # print('scale: ( Mean = {}, Var = {} )'.format(scale.mean().data[0], scale.var().data[0]))
        return scale


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=True):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer.forward(feat)
        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans


class CodePredictor(nn.Module):
    def __init__(self, nz_feat=100, n_kp=68):
        super(CodePredictor, self).__init__()
        self.quat_predictor = QuatPredictor(nz_feat)
        self.shape_predictor = ShapePredictor(nz_feat, n_kp=n_kp)
        self.scale_predictor = ScalePredictor(nz_feat)
        self.trans_predictor = TransPredictor(nz_feat)

    def forward(self, feat):
        """[summary]
        Returns:
            shape_pred.shape: torch.Size([1, 337, 3])
            scale_pred.shape: torch.Size([1, 1])
            quat_pred.shape: torch.Size([1, 4])
            trans_pred.shape: torch.Size([1, 2])
        """
        shape_pred = self.shape_predictor.forward(feat)
        scale_pred = self.scale_predictor.forward(feat)
        quat_pred = self.quat_predictor.forward(feat)
        trans_pred = self.trans_predictor.forward(feat)
        # print('shape_pred.shape:', shape_pred.shape)
        # print('scale_pred.shape:', scale_pred.shape)
        # print('quat_pred.shape:', quat_pred.shape)
        # print('trans_pred.shape:', trans_pred.shape)
        # raise ValueError
        return shape_pred, scale_pred, trans_pred, quat_pred

#------------- Network ------------#
#----------------------------------#
class IMM3DS(nn.Module):
    def __init__(self, nz_feat, n_kp, input_shape, mean_v_path=None, mean_v_fix=True):
        super(IMM3DS, self).__init__()
        self.encoder = Encoder(input_shape=input_shape, nz_feat=nz_feat)
        self.predictor = CodePredictor(nz_feat=nz_feat, n_kp=n_kp)
        self.nz_feat = nz_feat
        self.n_kp = n_kp
        self.input_shape = input_shape

        # load mean_v
        if mean_v_path is not None:
            self.mean_v = torch.Tensor(np.load(mean_v_path))
            if len(self.mean_v.shape) == 2:
                self.mean_v = self.mean_v.unsqueeze(0)
        else:
            self.mean_v = torch.rand(1, n_kp, 3)
        
        self.mean_v = nn.Parameter(self.mean_v)
        if mean_v_fix:
            self.mean_v.requires_grad = False


    def get_mean_shape(self):
        return self.mean_v

    def get_predictor(self, image):
        # image = ddata['image']
        code = self.encoder(image)
        delta_v, scale_pred, trans_pred, quat_pred = self.predictor(code)
        return delta_v, scale_pred, trans_pred, quat_pred

    def full_pipeline(self, ddata):
        delta_v1, scale_pred1, trans_pred1, quat_pred1 = self.get_predictor(ddata['img1'])
        cam_pred1 = torch.cat([scale_pred1, trans_pred1, quat_pred1], 1)

        delta_v2, scale_pred2, trans_pred2, quat_pred2 = self.get_predictor(ddata['img2'])
        cam_pred2 = torch.cat([scale_pred2, trans_pred2, quat_pred2], 1)

        mean_v = self.get_mean_shape()
        pred_v1 = mean_v + delta_v1
        pred_v2 = mean_v + delta_v2

        # proj to 2d
        pred_v_2d_1 = orthographic_proj_withz(pred_v1, cam_pred1)
        pred_v_2d_2 = orthographic_proj_withz(pred_v2, cam_pred2)

        res = {
            'pred_v1': pred_v1,
            'pred_v2': pred_v2,
            'cam_pred1': cam_pred1,
            'cam_pred2': cam_pred2,
            # 'delta_v': delta_v,
            # 'mean_v': mean_v,
            # 'pred_v_2d': pred_v_2d,
            # 'cam_pred': cam_pred,
            'pred_v_2d_1': pred_v_2d_1,
            'pred_v_2d_2': pred_v_2d_2
        }

        return res

    def forward(self, ddata, mode='full_pipeline'):
        if mode == 'full_pipeline':
            return self.full_pipeline(ddata)
        elif mode == 'get_predictor':
            return self.get_predictor(ddata['image'])
        else:
            raise ValueError('invalid mode `%s`' % mode)

# from .feature_extraction import FeatureExtracor

class IMM3D(nn.Module):
    def __init__(self, nz_feat, n_kp, input_shape, mean_v_path=None, mean_v_fix=True, \
        heatmap_size=16, heatmap_std=0.1, heatmap_mode='point', h_channel=32, lm_mean=None):
        super(IMM3D, self).__init__()
        self.encoder = Encoder(input_shape=input_shape, nz_feat=nz_feat)
        self.predictor = CodePredictor(nz_feat=nz_feat, n_kp=n_kp)
        self.nz_feat = nz_feat
        self.n_kp = n_kp
        self.input_shape = input_shape
        if lm_mean is None:
            lm_mean = [0, 0, 0]
        self.lm_mean = lm_mean

        self.heatmap_mode = heatmap_mode
        self.heatmap_size = heatmap_size
        self.heatmap_std = heatmap_std

        self.generater = MyGenerator(channels=n_kp+8*h_channel, h_channel=h_channel)
        self.content_encoder = MyContentEncoder(in_channel=3, h_channel=h_channel)

        self.ce = coordEnsymmicer()
        # self.fe = FeatureExtracor(networkname='resnet34')

        # load mean_v
        if mean_v_path is not None:
            self.mean_v = torch.Tensor((np.load(mean_v_path)-lm_mean) * 1.2)
            if len(self.mean_v.shape) == 2:
                self.mean_v = self.mean_v.unsqueeze(0)
        else:
            self.mean_v = torch.rand(1, n_kp, 3)
            # self.mean_v
        
        self.mean_v = nn.Parameter(self.mean_v)
        if mean_v_fix:
            self.mean_v.requires_grad = False

    def get_mean_shape(self):
        return self.mean_v

    def get_predictor(self, image):
        # image = ddata['image']
        code = self.encoder(image)
        delta_v, scale_pred, trans_pred, quat_pred = self.predictor(code)
        return delta_v, scale_pred, trans_pred, quat_pred

    def full_pipeline(self, ddata):
        delta_v, scale_pred, trans_pred, quat_pred = self.get_predictor(ddata['img'])
        cam_pred = torch.cat([scale_pred, trans_pred, quat_pred], 1)

        mean_v = self.get_mean_shape()
        pred_v = mean_v + delta_v

        # proj to 2d
        pred_v_2d = orthographic_proj(pred_v, cam_pred)

        res = {
            'delta_v': delta_v,
            'mean_v': mean_v,
            'pred_v': pred_v,
            'pred_v_2d': pred_v_2d,
            'cam_pred': cam_pred,
        }

        return res

    def equivariance_pipeline(self, ddata):
        img1 = ddata['img1']
        img2 = ddata['img2']
        mean_v = self.get_mean_shape()

        delta_v1, scale_pred1, trans_pred1, quat_pred1 = self.get_predictor(img1)
        cam_pred1 = torch.cat([scale_pred1, trans_pred1, quat_pred1], 1)
        # print('mean_v:', mean_v.shape, 'delta_v1:', delta_v1.shape)
        pred_v1 = mean_v + delta_v1
        pred_v_2d_1 = orthographic_proj(pred_v1, cam_pred1)

        delta_v2, scale_pred2, trans_pred2, quat_pred2 = self.get_predictor(img2)
        cam_pred2 = torch.cat([scale_pred2, trans_pred2, quat_pred2], 1)
        pred_v2 = mean_v + delta_v2
        pred_v_2d_2 = orthographic_proj(pred_v2, cam_pred2)

        res = {
            'pred_v_2d_1': pred_v_2d_1,
            'pred_v_2d_2': pred_v_2d_2,

            'delta_v1': delta_v1,
            'delta_v2': delta_v2,

            'cam_pred1': cam_pred1,
            'cam_pred2': cam_pred2
        }

        return res

    def IMM_pipeline(self, ddata):
        img1 = ddata['img1']
        img2 = ddata['img2']
        mean_v = self.get_mean_shape()

        delta_v1, scale_pred1, trans_pred1, quat_pred1 = self.get_predictor(img1)
        cam_pred1 = torch.cat([scale_pred1, trans_pred1, quat_pred1], 1)
        # print('mean_v:', mean_v.shape, 'delta_v1:', delta_v1.shape)
        pred_v1 = mean_v + delta_v1
        pred_v_2d_1 = orthographic_proj(pred_v1, cam_pred1)
        hm = get_gaussian_map_from_points(points=pred_v_2d_1, height=self.heatmap_size[0], weight=self.heatmap_size[1], std=self.heatmap_std, device=pred_v_2d_1.device, mode=self.heatmap_mode)

        content_2 = self.content_encoder(img2)
        feats = torch.cat((content_2, hm), dim=1)
        recovered_1 = self.generater(feats)

        res = {
            'recovered_1': recovered_1,
            'pred_v_2d_1': pred_v_2d_1,
            'pred_v1': pred_v1,
            'cam_pred1': cam_pred1,
            'delta_v1': delta_v1,
        }

        return res

    def IMM_pipeline_symmetric(self, ddata):
        img1 = ddata['img1']
        img2 = ddata['img2']
        mean_v = self.get_mean_shape()

        delta_v1, scale_pred1, trans_pred1, quat_pred1 = self.get_predictor(img1)
        cam_pred1 = torch.cat([scale_pred1, trans_pred1, quat_pred1], 1)
        # print('mean_v:', mean_v.shape, 'delta_v1:', delta_v1.shape)
        pred_v1 = mean_v + delta_v1
        pred_v_2d_1 = orthographic_proj(pred_v1, cam_pred1)
        hm = get_gaussian_map_from_points(points=pred_v_2d_1, height=self.heatmap_size[0], weight=self.heatmap_size[1], std=self.heatmap_std, device=pred_v_2d_1.device, mode=self.heatmap_mode)

        content_2 = self.content_encoder(img2)
        feats = torch.cat((content_2, hm), dim=1)
        recovered_1 = self.generater(feats)

        # symmetric part
        pred_v_2d_1_sym = self.ce(pred_v_2d_1)
        hm_sym = get_gaussian_map_from_points(points=pred_v_2d_1_sym, height=self.heatmap_size[0], weight=self.heatmap_size[1], std=self.heatmap_std, device=pred_v_2d_1_sym.device, mode=self.heatmap_mode)
        feats_sym = torch.cat((content_2, hm_sym), dim=1)
        recovered_1_sym = self.generater(feats_sym)
        recovered_1_sym = torch.flip(recovered_1_sym, (3,))

        # local feature

        res = {
            'recovered_1': recovered_1,
            'pred_v_2d_1': pred_v_2d_1,

            'pred_v1': pred_v1,
            'cam_pred1': cam_pred1,
            'delta_v1': delta_v1,

            'recovered_1_sym': recovered_1_sym,
            'pred_v_2d_1_sym': pred_v_2d_1_sym,
        }

        return res

    def IMM_pipeline_symmetric_SCOPSC(self, ddata):
        img1 = ddata['img1']
        img2 = ddata['img2']
        colormap = ddata['scops_mask'] # 9,16,16
        mean_v = self.get_mean_shape()

        delta_v1, scale_pred1, trans_pred1, quat_pred1 = self.get_predictor(img1)
        cam_pred1 = torch.cat([scale_pred1, trans_pred1, quat_pred1], 1)
        # print('mean_v:', mean_v.shape, 'delta_v1:', delta_v1.shape)
        pred_v1 = mean_v + delta_v1
        pred_v_2d_1 = orthographic_proj(pred_v1, cam_pred1) + torch.Tensor(self.lm_mean[:2]).to(pred_v1.device)
        hm = get_gaussian_map_from_points(points=pred_v_2d_1, height=self.heatmap_size[0], weight=self.heatmap_size[1], std=self.heatmap_std, device=pred_v_2d_1.device, mode=self.heatmap_mode) # B,68,16,16

        content_2 = self.content_encoder(img2)
        feats = torch.cat((content_2, hm), dim=1)
        recovered_1 = self.generater(feats)

        # symmetric part
        pred_v_2d_1_sym = self.ce(pred_v_2d_1)
        hm_sym = get_gaussian_map_from_points(points=pred_v_2d_1_sym, height=self.heatmap_size[0], weight=self.heatmap_size[1], std=self.heatmap_std, device=pred_v_2d_1_sym.device, mode=self.heatmap_mode)
        feats_sym = torch.cat((content_2, hm_sym), dim=1)
        recovered_1_sym = self.generater(feats_sym)
        recovered_1_sym = torch.flip(recovered_1_sym, (3,))

        # color map
        color_feat = (hm.unsqueeze(2) * colormap.unsqueeze(1)).sum((-1, -2)) # B,68,9

        # local feature

        res = {
            'recovered_1': recovered_1,
            'pred_v_2d_1': pred_v_2d_1,

            'pred_v1': pred_v1,
            'cam_pred1': cam_pred1,
            'delta_v1': delta_v1,

            'recovered_1_sym': recovered_1_sym,
            'pred_v_2d_1_sym': pred_v_2d_1_sym,

            'color_feat': color_feat,
        }

        return res


    def IMM_pipeline_symmetric_SCOPSC_visibility(self, ddata):
        img1 = ddata['img1']
        img2 = ddata['img2']
        colormap = ddata['scops_mask'] # 9,16,16
        mean_v = self.get_mean_shape()

        delta_v1, scale_pred1, trans_pred1, quat_pred1 = self.get_predictor(img1)
        cam_pred1 = torch.cat([scale_pred1, trans_pred1, quat_pred1], 1)
        # print('mean_v:', mean_v.shape, 'delta_v1:', delta_v1.shape)
        pred_v1 = mean_v + delta_v1

        # proj to 2d
        offset = self.lm_mean
        offset[-1] = 0
        pred_v_2d_1 = orthographic_proj_withz(pred_v1, cam_pred1) + torch.Tensor(offset).to(pred_v1.device)
        pred_v_z = pred_v_2d_1[...,2] # B,68,1,1
        pred_v_z_sign = (torch.sign(pred_v_z)+1) / 2

        # IMM map
        pred_v_2d_1 = pred_v_2d_1[..., :2]
        hm = get_gaussian_map_from_points(points=pred_v_2d_1, height=self.heatmap_size[0], weight=self.heatmap_size[1], std=self.heatmap_std, device=pred_v_2d_1.device, mode=self.heatmap_mode) # B,68,16,16
        hm = hm * pred_v_z_sign.unsqueeze(-1).unsqueeze(-1)

        content_2 = self.content_encoder(img2)
        feats = torch.cat((content_2, hm), dim=1)
        recovered_1 = self.generater(feats)

        # symmetric part
        pred_v_2d_1_sym = self.ce(pred_v_2d_1)
        hm_sym = get_gaussian_map_from_points(points=pred_v_2d_1_sym, height=self.heatmap_size[0], weight=self.heatmap_size[1], std=self.heatmap_std, device=pred_v_2d_1_sym.device, mode=self.heatmap_mode)
        pred_v_z_sign_sym = self.ce(pred_v_z_sign.unsqueeze(-1), flip=False).squeeze(-1)
        hm_sym = hm_sym * pred_v_z_sign_sym.unsqueeze(-1).unsqueeze(-1)
        feats_sym = torch.cat((content_2, hm_sym), dim=1)
        recovered_1_sym = self.generater(feats_sym)
        recovered_1_sym = torch.flip(recovered_1_sym, (3,))

        # color map
        color_feat = (hm.unsqueeze(2) * colormap.unsqueeze(1)).sum((-1, -2)) # B,68,9

        # local feature

        res = {
            'recovered_1': recovered_1,
            'pred_v_2d_1': pred_v_2d_1,

            'pred_v1': pred_v1,
            'cam_pred1': cam_pred1,
            'delta_v1': delta_v1,

            'recovered_1_sym': recovered_1_sym,
            'pred_v_2d_1_sym': pred_v_2d_1_sym,

            'color_feat': color_feat,
            'pred_v_z_sign': pred_v_z_sign,

            'hm': hm,
        }

        return res

    def IMM_pipeline_symmetric_localfeat(self, ddata):
        img1 = ddata['img1']
        img2 = ddata['img2']
        mean_v = self.get_mean_shape()

        delta_v1, scale_pred1, trans_pred1, quat_pred1 = self.get_predictor(img1)
        cam_pred1 = torch.cat([scale_pred1, trans_pred1, quat_pred1], 1)
        # print('mean_v:', mean_v.shape, 'delta_v1:', delta_v1.shape)
        pred_v1 = mean_v + delta_v1
        pred_v_2d_1 = orthographic_proj(pred_v1, cam_pred1)
        hm = get_gaussian_map_from_points(points=pred_v_2d_1, height=self.heatmap_size[0], weight=self.heatmap_size[1], std=self.heatmap_std, device=pred_v_2d_1.device, mode=self.heatmap_mode)

        content_2 = self.content_encoder(img2)
        feats = torch.cat((content_2, hm), dim=1)
        recovered_1 = self.generater(feats)

        # symmetric part
        pred_v_2d_1_sym = self.ce(pred_v_2d_1)
        hm_sym = get_gaussian_map_from_points(points=pred_v_2d_1_sym, height=self.heatmap_size[0], weight=self.heatmap_size[1], std=self.heatmap_std, device=pred_v_2d_1_sym.device, mode=self.heatmap_mode)
        feats_sym = torch.cat((content_2, hm_sym), dim=1)
        recovered_1_sym = self.generater(feats_sym)
        recovered_1_sym = torch.flip(recovered_1_sym, (3,))

        # local feature
        # hm = get_gaussian_map_from_points(pred_v_2d_1, 16, 16, 0.02, device=pred_v_2d_1.device, mode='point')
        feat = self.fe(norm(img1))
        local_feat = [(hm.unsqueeze(2) * ii.unsqueeze(1)).sum((-1, -2)) for ii in feat]


        res = {
            'recovered_1': recovered_1,
            'pred_v_2d_1': pred_v_2d_1,

            'pred_v1': pred_v1,
            'cam_pred1': cam_pred1,
            'delta_v1': delta_v1,

            'recovered_1_sym': recovered_1_sym,
            'pred_v_2d_1_sym': pred_v_2d_1_sym,

            'local_feat': local_feat,
        }

        return res

    def forward(self, ddata, mode='full_pipeline'):
        if mode == 'full_pipeline':
            return self.IMM_pipeline_symmetric_SCOPSC_visibility(ddata)
        elif mode == 'IMM_pipeline_symmetric':
            return self.IMM_pipeline_symmetric(ddata)
        elif mode == 'get_predictor':
            return self.get_predictor(ddata['image'])
        else:
            raise ValueError('invalid mode `%s`' % mode)


class IMM3DS(nn.Module):
    def __init__(self, nz_feat, n_kp, input_shape, n_lm=10, mean_v_path=None, mean_v_fix=True, \
        heatmap_size=16, heatmap_std=0.1, heatmap_mode='point', h_channel=32, lm_mean=None):
        super(IMM3DS, self).__init__()
        self.encoder = Encoder(input_shape=input_shape, nz_feat=nz_feat)
        self.predictor = CodePredictor(nz_feat=nz_feat, n_kp=n_kp)
        self.pose_encoder = MyPoseEncoder(dim=n_kp, heatmap_std=heatmap_std, in_channel=3, h_channel=h_channel, heatmap_size=heatmap_size, mode=heatmap_mode)

        self.nz_feat = nz_feat
        self.n_kp = n_kp
        self.n_lm = n_lm
        self.input_shape = input_shape
        if lm_mean is None:
            lm_mean = [0, 0, 0]
        self.lm_mean = lm_mean

        self.heatmap_mode = heatmap_mode
        self.heatmap_size = heatmap_size
        self.heatmap_std = heatmap_std

        self.generater = MyGenerator(channels=n_kp+8*h_channel, h_channel=h_channel)
        self.content_encoder = MyContentEncoder(in_channel=3, h_channel=h_channel)

        self.ce = coordEnsymmicer()
        # self.fe = FeatureExtracor(networkname='resnet34')
        # self.kp2lm = nn.Linear()

        # load mean_v
        if mean_v_path is not None:
            self.mean_v = torch.Tensor((np.load(mean_v_path)-lm_mean) * 1.2)
            if len(self.mean_v.shape) == 2:
                self.mean_v = self.mean_v.unsqueeze(0)
        else:
            self.mean_v = torch.rand(1, n_kp, 3)
            # self.mean_v
        
        self.mean_v = nn.Parameter(self.mean_v)
        if mean_v_fix:
            self.mean_v.requires_grad = False

    def get_mean_shape(self):
        return self.mean_v

    def get_predictor(self, image):
        # image = ddata['image']
        code = self.encoder(image)
        delta_v, scale_pred, trans_pred, quat_pred = self.predictor(code)
        return delta_v, scale_pred, trans_pred, quat_pred

    def IMM_pipeline_symmetric_SCOPSC_visibility_separate(self, ddata):
        # prepare data
        image = ddata['image']
        img1 = ddata['img1']
        img2 = ddata['img2']
        colormap = ddata['scops_mask'] # 9,16,16
        mean_v = self.get_mean_shape()

        # parameter estimation
        delta_v, scale_pred, trans_pred, quat_pred = self.get_predictor(image)
        cam_pred = torch.cat([scale_pred, trans_pred, quat_pred], 1)
        # print('mean_v:', mean_v.shape, 'delta_v1:', delta_v1.shape)
        pred_v = mean_v + delta_v

        # proj to 2d
        offset = self.lm_mean
        offset[-1] = 0
        pred_v_2d = orthographic_proj_withz(pred_v, cam_pred) + torch.Tensor(offset).to(pred_v.device)
        pred_v_z = pred_v_2d[...,2] # B,68
        pred_v_z_sign = (torch.sign(pred_v_z)+1) / 2
        pred_v_2d = pred_v_2d[..., :2]
        hm, lm, _ = self.pose_encoder(image)


        # IMM pipeline
        hm_y, lm_y, hmr_y= self.pose_encoder(img1)
        # hm = get_gaussian_map_from_points(points=pred_v_2d_1, height=self.heatmap_size[0], weight=self.heatmap_size[1], std=self.heatmap_std, device=pred_v_2d_1.device, mode=self.heatmap_mode) # B,68,16,16
        hm_y = hm_y * pred_v_z_sign.unsqueeze(-1).unsqueeze(-1)
        content_2 = self.content_encoder(img2)
        feats = torch.cat((content_2, hm_y), dim=1)
        recovered_1 = self.generater(feats)

        # symmetric part
        lm_y_sym = self.ce(lm_y)
        hm_sym = get_gaussian_map_from_points(points=lm_y_sym, height=self.heatmap_size[0], weight=self.heatmap_size[1], std=self.heatmap_std, device=lm_y_sym.device, mode=self.heatmap_mode)
        pred_v_z_sign_sym = self.ce(pred_v_z_sign.unsqueeze(-1), flip=False).squeeze(-1)
        hm_sym = hm_sym * pred_v_z_sign_sym.unsqueeze(-1).unsqueeze(-1)
        feats_sym = torch.cat((content_2, hm_sym), dim=1)
        recovered_1_sym = self.generater(feats_sym)
        recovered_1_sym = torch.flip(recovered_1_sym, (3,))

        # color map
        color_feat = (hm_y.unsqueeze(2) * colormap.unsqueeze(1)).sum((-1, -2)) # B,68,9

        res = {
            # for structure loss
            'pred_v_2d': pred_v_2d,
            'pred_v_z': pred_v_z,
            'pred_v_z_sign': pred_v_z_sign,
            'pred_v_z_sign_sym': pred_v_z_sign_sym,
            'lm': lm,

            # for equivariance loss
            'recovered_1': recovered_1,
            'recovered_1_sym': recovered_1_sym,
            'lm_y': lm_y,
            'lm_y_sym': lm_y_sym,

            # camera
            'pred_v': pred_v,
            'cam_pred': cam_pred,
            'delta_v': delta_v,
            
            # for SCOPS color
            'color_feat': color_feat,

            # color loss
            'hm': hm,
        }
        return res

    def forward(self, ddata, mode='full_pipeline'):
        if mode == 'full_pipeline':
            return self.IMM_pipeline_symmetric_SCOPSC_visibility_separate(ddata)
        elif mode == 'get_predictor':
            return self.get_predictor(ddata['image'])
        else:
            raise ValueError('invalid mode `%s`' % mode)


class coordEnsymmicer():
    def __init__(self):
        self.map68 = {}
        self.map68.update({i:16-i for i in range(17)})
        self.map68.update({i:43-i for i in range(17, 27)}) # 17: 26
        self.map68.update({i:i for i in range(27, 31)}) #
        self.map68.update({i:66-i for i in range(31, 36)}) # 31: 35
        self.map68.update({i:81-i for i in range(36, 40)}) # 36:45
        self.map68.update({i:87-i for i in range(40, 42)}) # 40: 47
        self.map68.update({i:81-i for i in range(42, 46)}) # 36:45
        self.map68.update({i:87-i for i in range(46, 48)}) # 46: 41 
        self.map68.update({i:102-i for i in range(48, 55)}) # 48: 54
        self.map68.update({i:114-i for i in range(55, 60)}) # 55: 59
        self.map68.update({i:124-i for i in range(60, 65)}) # 60: 64
        self.map68.update({i:132-i for i in range(65, 68)}) # 65: 67

        self.matrix = torch.zeros(1, 68, 68)
        for k,v in self.map68.items():
            self.matrix[0, k, v] = 1

    def __call__(self, coord, flip=True):
        coord_new = self.matrix.repeat(coord.size(0), 1, 1).to(coord.device).bmm(coord)
        if flip:
            coord_new[:, :, 0] = 1 - coord_new[:, :, 0]
        return coord_new



if __name__ == "__main__":
    x = torch.rand(10, 3, 128, 128)
    net = Encoder(input_shape=(128, 128))
    print(net(x).shape)