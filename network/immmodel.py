from network.feature_extraction import FeatureExtraction
from torchvision import models
from imgtransform.trans_controller import get_trans_controller
from utils.utils import gifdict, squeeze_recur
from utils.soft_points import \
    get_expected_points_from_map, \
    get_gaussian_map_from_points
import torch
import torch.nn as nn


class HeatMap(nn.Module):
    """
    Refine the estimated pose map to be gaussian distributed heatmap.
    Calculate the gaussian mean value.
    Params:
    std: standard deviation of gaussian distribution
    output_size: output feature map size
    """

    def __init__(self, std, output_size, mode='rot'):
        super(HeatMap, self).__init__()
        self.std = std
        self.out_h, self.out_w = output_size
        self.mode = mode

    def forward(self, x):
        """
        x: feature map BxCxHxW
        """
        B,C,H,W = x.shape
        coord = get_expected_points_from_map(x)
        res = get_gaussian_map_from_points(coord, H, W, self.std, x.device, mode=self.mode)
        return res, coord


class Conv_Block_D(nn.Module):
    def __init__(self, inc, outc, downsample=False):
        super(Conv_Block_D, self).__init__()
        if downsample:
            layer_i = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=4, stride=2, padding=1)
        else:
            layer_i = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, padding=1)
        block = [
            layer_i,
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


# class Conv_Block_G(nn.Module):
#     def __init__(self, inc, outc, upsample=False):
#         super(Conv_Block_G, self).__init__()
#         if upsample:
#             layer_i = nn.ConvTranspose2d(in_channels=inc, out_channels=outc, stride=2, kernel_size=4, padding=1)
#         else:
#             layer_i = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, padding=1)
#         block = [
#             layer_i,
#             nn.BatchNorm2d(outc),
#             nn.ReLU(inplace=True)
#         ]
#         self.block = nn.Sequential(*block)

#     def forward(self, x):
#         return self.block(x)

class Conv_Block_G(nn.Module):
    def __init__(self, inc, outc, upsample=False):
        super(Conv_Block_G, self).__init__()
        
        block = [
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        ]
        if upsample:
            block.append(nn.Upsample(size=(128, 128), mode='bilinear'))
        self.block = nn.Sequential(*block)
        

    def forward(self, x):
        return self.block(x)


class MyContentEncoder(nn.Module):
    def __init__(self, in_channel=3, h_channel=64):
        super(MyContentEncoder, self).__init__()
        self.conv1_1 = Conv_Block_D(in_channel, h_channel)
        self.conv1_2 = Conv_Block_D(h_channel, h_channel)

        self.conv2_1 = Conv_Block_D(h_channel, 2 * h_channel, downsample=True)
        self.conv2_2 = Conv_Block_D(2 * h_channel, 2 * h_channel)

        self.conv3_1 = Conv_Block_D(2 * h_channel, 4 * h_channel, downsample=True)
        self.conv3_2 = Conv_Block_D(4 * h_channel, 4 * h_channel)

        self.conv4_1 = Conv_Block_D(4 * h_channel, 8 * h_channel, downsample=True)
        self.conv4_2 = Conv_Block_D(8 * h_channel, 8 * h_channel)

        self.conv_layers = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv4_1,
            self.conv4_2
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class MyPoseEncoder(nn.Module):
    def __init__(self, dim=10, heatmap_std=0.1, in_channel=3, h_channel=32, heatmap_size=16, mode='point'):
        """

        Args:
            dim (int): Num of keypoints
        """
        super(MyPoseEncoder, self).__init__()
        self.conv1_1 = Conv_Block_D(in_channel, h_channel)
        self.conv1_2 = Conv_Block_D(h_channel, h_channel)

        self.conv2_1 = Conv_Block_D(h_channel, 2 * h_channel, downsample=True)
        self.conv2_2 = Conv_Block_D(2 * h_channel, 2 * h_channel)

        self.conv3_1 = Conv_Block_D(2 * h_channel, 4 * h_channel, downsample=True)
        self.conv3_2 = Conv_Block_D(4 * h_channel, 4 * h_channel)

        self.conv4_1 = Conv_Block_D(4 * h_channel, 8 * h_channel, downsample=True)
        self.conv4_2 = Conv_Block_D(8 * h_channel, 8 * h_channel)

        self.out_conv = nn.Conv2d(8 * h_channel, dim, (1, 1))
        self.conv_layers = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv4_1,
            self.conv4_2,
        ])

        self.mode = mode
        self.heatmap = HeatMap(heatmap_std, (heatmap_size, heatmap_size), mode=mode)
        
    def forward(self, x, return_feat=False):
        for layer in self.conv_layers:
            x = layer(x)
        if return_feat:
            h = x                               # B,256,16,16
        x = self.out_conv(x)                    # B,C,16,16 raw_heatmap
        heatmap, landmark = self.heatmap(x)     # B,C,16,16; B,C,2
        if return_feat:
            return heatmap, landmark, x, h
        return heatmap, landmark, x


class MyGenerator(nn.Module):
    """"""

    def __init__(self, channels=64 + 10, h_channel=32):
        super(MyGenerator, self).__init__()
        self.conv1_1 = Conv_Block_G(channels, 8 * h_channel)
        self.conv1_2 = Conv_Block_G(8 * h_channel, 8 * h_channel, upsample=True)

        self.conv2_1 = Conv_Block_G(8 * h_channel, 4 * h_channel)
        self.conv2_2 = Conv_Block_G(4 * h_channel, 4 * h_channel, upsample=True)

        self.conv3_1 = Conv_Block_G(4 * h_channel, 2 * h_channel)
        self.conv3_2 = Conv_Block_G(2 * h_channel, 2 * h_channel, upsample=True)

        self.conv4_1 = Conv_Block_G(2 * h_channel, h_channel)

        self.final_conv = nn.Conv2d(h_channel, 3, (3, 3), padding=[1, 1])

        self.conv_layers = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv4_1,
            self.final_conv
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class MyIMM(nn.Module):
    def __init__(self, dim=10, 
            heatmap_std=0.1, 
            in_channel=3, 
            h_channel=32, 
            item_map=dict(x='x', y='y', img='img'),
            mode='point',
            **kw):
        """
        It should be noted all params has been fixed to Jakab 2018 paper.
        Goto the original class if params and layers need to be changed.
        Images should be rescaled to 128*128
        """
        super(MyIMM, self).__init__()
        self.content_encoder = MyContentEncoder(in_channel, h_channel)
        self.pose_encoder = MyPoseEncoder(dim, heatmap_std, in_channel, h_channel, mode=mode)
        self.generator = MyGenerator(channels=8*h_channel+dim, h_channel=h_channel)

        self.item_map = item_map

    def full_pipeline(self, x, y):
        """
        x,y: B X C X H X W (B,3,128,128)
        """
        content_x = self.content_encoder(x) # ([B, 256, 16, 16])
        hm_y, lm_y, hmr_y= self.pose_encoder(y)
        code = torch.cat((content_x, hm_y), dim=1) # ([B, 266, 16, 16])
        recovered_y = self.generator(code) 

        res = {
            'recovered_y': recovered_y,
            'hm_y': hm_y,
            'lm_y': lm_y,
            'hmr_y': hmr_y
        }
        return res

    def get_lm_pred(self, img):
        # img = ddata['image']
        hm_y, lm_y, hmr_y = self.pose_encoder(img) # ([B, 10, 16, 16]), ([B, 10, 2])
        res = {
            'hm_y': hm_y,
            'lm_y': lm_y,
            'hmr_y': hmr_y
        }
        return res

    def forward(self, ddata, mode='full_pipeline'):
        """[summary]

        Args:
            ddata ([type]): dict of data. 
                function and data corresponding to: 
                {full_pipeline: x, y} {get_lm_pred: img}
            mode (str, optional): [description]. Defaults to 'full_pipeline'.
        """
        if mode == 'full_pipeline':
            return self.full_pipeline(ddata['img1'], ddata['img2'])
        elif mode == 'get_lm_pred':
            return self.get_lm_pred(ddata['img'])
        else:
            raise ValueError('invalid mode `%s`' % mode)



class MyIMMPP(nn.Module):
    def __init__(self, dim=10, 
            heatmap_std=0.1, 
            in_channel=3, 
            h_channel=32, 
            item_map=dict(x='x', y='y', img='img'),
            mode='point',
            **kw):
        """
        It should be noted all params has been fixed to Jakab 2018 paper.
        Goto the original class if params and layers need to be changed.
        Images should be rescaled to 128*128
        """
        super(MyIMMPP, self).__init__()
        self.content_encoder = MyContentEncoder(in_channel, h_channel)
        self.pose_encoder = MyPoseEncoder(dim, heatmap_std, in_channel, h_channel, mode=mode)
        self.generator = MyGenerator(channels=8*h_channel+dim, h_channel=h_channel)

        self.conv_pred = nn.Conv2d(8 * h_channel, dim, (1, 1))
        self.pool_pred = nn.AdaptiveAvgPool2d((1,1))

        self.item_map = item_map

    def full_pipeline(self, x, y):
        """
        x,y: B X C X H X W (B,3,128,128)
        """
        content_x = self.content_encoder(x) # ([B, 256, 16, 16])
        hm_y, lm_y, hmr_y, pb_y = self.pose_encoder(y, True)

        pb_y = self.conv_pred(pb_y) # B,10,16,16
        pb_y = self.pool_pred(pb_y) # B,10,1,1,
        # pb_y = torch.nn.functional.relu(pb_y)
        # pb_y = torch.nn.functional.softmax(pb_y, -1)
        # pb_y = torch.clamp(pb_y, 0, 1)
        pb_y = torch.nn.functional.sigmoid(pb_y)
        # pb_y[pb_y>0] = 1
        # pb_y[pb_y<0] = 0

        # print('pb_y.shape:', pb_y.shape)
        hm_y = hm_y * pb_y
        # print('hm_y.shape:', hm_y.shape)
        # print('content_x.shape:', content_x.shape)
        code = torch.cat((content_x, hm_y), dim=1) # ([B, 266, 16, 16])
        recovered_y = self.generator(code) 

        res = {
            'recovered_y': recovered_y,
            'hm_y': hm_y,
            'lm_y': lm_y,
            'hmr_y': hmr_y,
            'pb_y': pb_y,
        }
        return res

    def get_lm_pred(self, img):
        # img = ddata['image']
        hm_y, lm_y, hmr_y, pb_y = self.pose_encoder(img, True) # ([B, 10, 16, 16]), ([B, 10, 2])

        pb_y = self.conv_pred(pb_y)
        pb_y = self.pool_pred(pb_y)
        # pb_y = torch.nn.functional.relu(pb_y)
        pb_y = torch.nn.functional.softmax(pb_y, -1)
        # pb_y = torch.nn.functional.sigmoid(pb_y)
        # pb_y = torch.clamp(pb_y, 0, 1)
        # pb_y[pb_y>0] = 1
        # pb_y[pb_y<0] = 0


        res = {
            'hm_y': hm_y,
            'lm_y': lm_y,
            'hmr_y': hmr_y,
            'pb_y': pb_y
        }
        return res

    def forward(self, ddata, mode='full_pipeline'):
        """[summary]

        Args:
            ddata ([type]): dict of data. 
                function and data corresponding to: 
                {full_pipeline: x, y} {get_lm_pred: img}
            mode (str, optional): [description]. Defaults to 'full_pipeline'.
        """
        if mode == 'full_pipeline':
            return self.full_pipeline(ddata['img1'], ddata['img2'])
        elif mode == 'get_lm_pred':
            return self.get_lm_pred(ddata['img'])
        else:
            raise ValueError('invalid mode `%s`' % mode)




def norm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(img.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(img.device)
    img = (img.permute(0, 2, 3, 1) - mean) / std
    return img.permute(0, 3, 1, 2)

class IMMSC(nn.Module):
    """imm with semantic consistency
    """
    def __init__(self, dim=10, 
            heatmap_std=0.1, 
            in_channel=3, 
            h_channel=32, 
            item_map=dict(x='x', y='y', img='img'),
            mode='point',
            fenetname = 'vgg19',
            normalization=False,
            layers = 'relu5_2,relu5_4',
            output_size=(16, 16),
            **kw):
        """
        It should be noted all params has been fixed to Jakab 2018 paper.
        Goto the original class if params and layers need to be changed.
        Images should be rescaled to 128*128
        """
        super(IMMSC, self).__init__()
        self.content_encoder = MyContentEncoder(in_channel, h_channel)
        self.pose_encoder = MyPoseEncoder(dim, heatmap_std, in_channel, h_channel, mode=mode)
        self.generator = MyGenerator(channels=8*h_channel+dim, h_channel=h_channel)
        # self.feature_extractor = FeatureExtracor(networkname=fenetname)
        self.feature_extractor = FeatureExtraction(train_fe=False, feature_extraction_cnn=fenetname, normalization=normalization, last_layer=layers, use_cuda=False, output_size=output_size)

        vec = torch.rand(1, dim, 512)
        self.feat_vec = nn.Parameter(vec)

        self.item_map = item_map

    def full_pipeline(self, x, y):
        """
        x,y: B X C X H X W (B,3,128,128)
        """
        content_x = self.content_encoder(x) # ([B, 256, 16, 16])
        hm_y, lm_y, hmr_y= self.pose_encoder(y)
        code = torch.cat((content_x, hm_y), dim=1) # ([B, 266, 16, 16])
        recovered_y = self.generator(code) 

        # local feat
        # hmr_y = hmr_y - hmr_y.min(2, True)[0].min(3, True)[0] / hmr_y.max(2, True)[0].max(3, True)[0] - hmr_y.min(2, True)[0].min(3, True)[0]
        # hm_y = hm_y / hmr_y.max(2, True)[0].max(3, True)[0] - hmr_y.min(2, True)[0].min(3, True)[0]
        fe_y = self.feature_extractor(norm(y))
        fe_y = torch.cat(fe_y, dim=1)
        local_feat = (hm_y.unsqueeze(2) * fe_y.unsqueeze(1)).sum((-1, -2))
        # local_feat = local_feat / local_feat.sum(-1, keepdim=True) * local_feat.shape[-1]

        # # hm_y: B,10,16,16;  fe_y: B,960,16,16
        # local_feat = [(hmr_y.unsqueeze(2) * ii.unsqueeze(1)).sum((-1, -2)) for ii in fe_y] # B,10,960
        # local_feat = [(hm_y.unsqueeze(2) * ii.unsqueeze(1)).sum((-1, -2)) for ii in fe_y] # B,10,960

        res = {
            'recovered_y': recovered_y,
            'hm_y': hm_y,
            'lm_y': lm_y,
            'hmr_y': hmr_y,
            'local_feat': local_feat,
            'fix_feat': self.feat_vec
        }
        return res

    def get_lm_pred(self, img):
        # img = ddata['image']
        hm_y, lm_y, hmr_y = self.pose_encoder(img) # ([B, 10, 16, 16]), ([B, 10, 2])
        res = {
            'hm_y': hm_y,
            'lm_y': lm_y,
            'hmr_y': hmr_y
        }
        return res

    def forward(self, ddata, mode='full_pipeline'):
        """[summary]

        Args:
            ddata ([type]): dict of data. 
                function and data corresponding to: 
                {full_pipeline: x, y} {get_lm_pred: img}
            mode (str, optional): [description]. Defaults to 'full_pipeline'.
        """
        if mode == 'full_pipeline':
            return self.full_pipeline(ddata['img1'], ddata['img2'])
        elif mode == 'get_lm_pred':
            return self.get_lm_pred(ddata['img'])
        else:
            raise ValueError('invalid mode `%s`' % mode)

class MyIMMBN(nn.Module):
    def __init__(self, dim=10, 
            heatmap_std=0.1, 
            in_channel=3, 
            h_channel=32, 
            item_map=dict(x='x', y='y', img='img'),
            mode='point',
            **kw):
        """
        IMM 
        """
        super(MyIMMBN, self).__init__()
        self.content_encoder = MyContentEncoder(in_channel, h_channel)
        self.pose_encoder = MyPoseEncoder(dim, heatmap_std, in_channel, h_channel, mode=mode)
        self.generator = MyGenerator(channels=8*h_channel, h_channel=h_channel)

        self.item_map = item_map

    def full_pipeline(self, x, y):
        """
        x,y: B X C X H X W (B,3,128,128)
        """
        content_x = self.content_encoder(x) # ([B, 256, 16, 16])
        hm_x, lm_x, hmr_x = self.pose_encoder(x)
        lf_x = (hm_x.unsqueeze(2) * content_x.unsqueeze(1)).sum((-1, -2), keepdim=True) # B,10,256,1,1
        hm_y, lm_y, hmr_y= self.pose_encoder(y)

        code = (lf_x * hm_y.unsqueeze(2)).sum(1) # B,256,16,16
        recovered_y = self.generator(code) 

        res = {
            'recovered_y': recovered_y,
            'hm_y': hm_y,
            'lm_y': lm_y,
            'hmr_y': hmr_y
        }
        return res

    def get_lm_pred(self, img):
        # img = ddata['image']
        hm_y, lm_y, hmr_y = self.pose_encoder(img) # ([B, 10, 16, 16]), ([B, 10, 2])
        res = {
            'hm_y': hm_y,
            'lm_y': lm_y,
            'hmr_y': hmr_y
        }
        return res

    def forward(self, ddata, mode='full_pipeline'):
        """[summary]

        Args:
            ddata ([type]): dict of data. 
                function and data corresponding to: 
                {full_pipeline: x, y} {get_lm_pred: img}
            mode (str, optional): [description]. Defaults to 'full_pipeline'.
        """
        if mode == 'full_pipeline':
            return self.full_pipeline(ddata['img1'], ddata['img2'])
        elif mode == 'get_lm_pred':
            return self.get_lm_pred(ddata['img'])
        else:
            raise ValueError('invalid mode `%s`' % mode)