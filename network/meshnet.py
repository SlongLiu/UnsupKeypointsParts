from geoutils.geom_utils import save_objfile
from network.nmr import NeuralRenderer
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

from geoutils import mesh
import network.net_blocks as nb


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
            x = self.resnet.layer4(x)
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


class TexturePredictorUV(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, nz_feat, uv_sampler, opts, img_H=64, img_W=128, n_upconv=5, nc_init=256, predict_flow=False, symmetric=False, num_sym_faces=624):
        super(TexturePredictorUV, self).__init__()
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.symmetric = symmetric
        self.num_sym_faces = num_sym_faces
        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)
        self.predict_flow = predict_flow
        # B x F x T x T x 2 --> B x F x T*T x 2
        self.uv_sampler = uv_sampler.view(-1, self.F, self.T*self.T, 2)

        self.enc = nb.fc_stack(nz_feat, self.nc_init*self.feat_H*self.feat_W, 2)
        if predict_flow:
            nc_final=2
        else:
            nc_final=3
        self.decoder = nb.decoder2d(n_upconv, None, nc_init, init_fc=False, nc_final=nc_final, use_deconv=opts.use_deconv, upconv_mode=opts.upconv_mode)

    def forward(self, feat):
        # pdb.set_trace()
        uvimage_pred = self.enc.forward(feat)
        uvimage_pred = uvimage_pred.view(uvimage_pred.size(0), self.nc_init, self.feat_H, self.feat_W)
        # B x 2 or 3 x H x W
        self.uvimage_pred = self.decoder.forward(uvimage_pred)
        self.uvimage_pred = torch.tanh(self.uvimage_pred)

        tex_pred = torch.nn.functional.grid_sample(self.uvimage_pred, self.uv_sampler, align_corners=True)
        tex_pred = tex_pred.view(uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)

        if self.symmetric:
            # Symmetrize.
            tex_left = tex_pred[:, -self.num_sym_faces:]
            return torch.cat([tex_pred, tex_left], 1)
        else:
            # Contiguous Needed after the permute..
            return tex_pred.contiguous()



class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, num_verts):
        super(ShapePredictor, self).__init__()
        # self.pred_layer = nb.fc(True, nz_feat, num_verts)
        self.pred_layer = nn.Linear(nz_feat, num_verts * 3)

        # Initialize pred_layer weights to be small so initial def aren't so big
        self.pred_layer.weight.data.normal_(0, 0.0001)

    def forward(self, feat):
        # pdb.set_trace()
        delta_v = self.pred_layer.forward(feat)
        # Make it B x num_verts x 3
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
    def __init__(self, nz_feat=100, num_verts=1000):
        super(CodePredictor, self).__init__()
        self.quat_predictor = QuatPredictor(nz_feat)
        self.shape_predictor = ShapePredictor(nz_feat, num_verts=num_verts)
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



#------------ Mesh Net ------------#
#----------------------------------#
class MeshNet(nn.Module):
    def __init__(self, input_shape, nz_feat=100, num_kps=10, symmetric=True, args=None, **kw):
        super(MeshNet, self).__init__()
        self.input_shape = input_shape
        self.nz_feat = nz_feat
        self.num_kps = num_kps
        self.symmetric = symmetric
        self.args = args

        # get mean shape (sphere)
        verts, faces = mesh.create_sphere(args.subdivde)
        # save_objfile(verts, faces, 'tmp/face.obj')
        # print('verts.shape:', verts.shape) # 642, 3
        # print('faces.shape:', faces.shape) # 1280, 3
        # raise ValueError
        num_verts = verts.shape[0]
        self.num_verts = num_verts

        if self.symmetric:
            verts, faces, num_indept, num_sym, num_indept_faces, num_sym_faces = mesh.make_symmetric(verts, faces)
            num_sym_output = num_indept + num_sym

            self.num_indept = num_indept
            self.num_sym = num_sym
            self.num_indept_faces = num_indept_faces
            self.num_sym_faces = num_sym_faces
            self.num_sym_output = num_sym_output

            # from other code
            self.num_output = num_sym_output

            # mean_v to be updated
            self.mean_v = nn.Parameter(torch.Tensor(verts[:num_sym_output]))

            # Needed for symmetrizing..
            self.flip = Variable(torch.ones(1, 3).cuda(), requires_grad=False)
            self.flip[0, 0] = -1
        else:
            self.mean_v = nn.Parameter(torch.Tensor(verts))
            self.num_output = self.num_verts

        verts_np = verts
        faces_np = faces
        self.faces = Variable(torch.LongTensor(faces).cuda(), requires_grad=False)
        self.edges2verts = mesh.compute_edges2verts(verts, faces)

        # initial vert2kp (after softmax)
        vert2kp_init = torch.Tensor(np.ones((num_kps, num_verts)) / float(num_verts))
        self.vert2kp_init = torch.nn.functional.softmax(Variable(vert2kp_init.cuda(), requires_grad=False), dim=1)
        self.vert2kp = nn.Parameter(vert2kp_init) # n_kps X n_verts

        # encoder
        self.encoder = Encoder(input_shape, n_blocks=4, nz_feat=nz_feat)
        self.code_predictor = CodePredictor(nz_feat=nz_feat, num_verts=self.num_output)

    def forward(self, sample):
        img = sample['image']
        img_feat = self.encoder.forward(img)
        shape_pred, scale_pred, trans_pred, quat_pred = self.code_predictor.forward(img_feat)
        # res = {
        #     'shape_pred': shape_pred,
        #     'scale_pred': scale_pred,
        #     'trans_pred': trans_pred,
        #     'quat_pred': quat_pred
        # }
        return shape_pred, scale_pred, trans_pred, quat_pred

    def symmetrize(self, V):
        if self.symmetric:
            if V.dim() == 2:
                V_left = self.flip * V[-self.num_sym:]
                return torch.cat([V, V_left], 0)
            else:
                V_left = self.flip * V[:, -self.num_sym:]
                return torch.cat([V, V_left], 1)
        else:
            return V

    def get_mean_shape(self):
        return self.symmetrize(self.mean_v)


#------------ Warper --------------#
#----------------------------------#
class MeshNetPlus(nn.Module):
    def __init__(self, args, **kw):
        super(MeshNetPlus, self).__init__()
        self.meshnet = MeshNet(args=args, **kw)
        self.render = NeuralRenderer(img_size=kw['input_shape'][0])

        # get faces
        faces = self.meshnet.faces.view(1, -1, 3)
        self.faces = faces.repeat(args.batch_size, 1, 1)
        # self.faces = faces

        # counter
        self.cnt = 0

    def full_pipeline(self, ddata):
        # img, saliency, kp, kp_prob
        delta_v, scale_pred, trans_pred, quat_pred = self.meshnet(ddata)
        cam_pred = torch.cat([scale_pred, trans_pred, quat_pred], 1)

        # print('delta_v:', delta_v.shape)
        # print('cam_pred:', cam_pred.shape)
        
        # get pred_v 
        delta_v = self.meshnet.symmetrize(delta_v)
        mean_v = self.meshnet.get_mean_shape()
        pred_v = mean_v + delta_v

        vert2kp = torch.nn.functional.softmax(self.meshnet.vert2kp, dim=1)
        kp_verts = torch.matmul(vert2kp, pred_v) # kp landmarks, out:B,N,3

        proj_cam = cam_pred

        # projection
        kp_pred = self.render.project_points(kp_verts, proj_cam)
        if delta_v.size(0) == 1:
            faces_used = self.faces[0].unsqueeze(0)
        else:
            faces_used = self.faces
        # faces_used = self.faces.repeat((delta_v.size(0), 1, 1))
        mask_pred = self.render(pred_v, faces_used, proj_cam)

        res = {
            'vert2kp': vert2kp,
            'delta_v': delta_v,
            'pred_v': pred_v,
            'kp_pred': kp_pred,
            'mask_pred': mask_pred,
            'cam_pred': cam_pred,
        }

        return res

    def forward(self, sample, mode='full_pipeline'):
        """[summary]

        Args:
            sample ([type]): [description]
            mode (str, optional): 'full_pipeline' or 'get_mean_shape'. Defaults to 'full_pipeline'.
        """
        if mode == 'get_mean_shape':
            return self.meshnet.get_mean_shape()

        if mode == 'full_pipeline':
            return self.full_pipeline(sample)


