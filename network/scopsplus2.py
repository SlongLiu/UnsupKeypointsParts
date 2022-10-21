from utils.soft_points import get_expected_points_from_map, get_sigma
from network.immmodel import HeatMap
import torch
import torch.nn as nn
from itertools import chain 

from .resnet import resnet34, resnet18, resnet50


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
            nn.LeakyReLU(inplace=True)
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class Conv_Block_G(nn.Module):
    def __init__(self, inc, outc, upsample=False):
        super(Conv_Block_G, self).__init__()
        if upsample:
            layer_i = nn.ConvTranspose2d(in_channels=inc, out_channels=outc, stride=2, kernel_size=4, padding=1)
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


class MyGenerator(nn.Module):
    """"""

    def __init__(self, channels=64 + 10, h_channel=32, downsamplelast=False):
        super(MyGenerator, self).__init__()
        self.conv1_1 = Conv_Block_G(channels, 8 * h_channel)
        self.conv1_2 = Conv_Block_G(8 * h_channel, 8 * h_channel, upsample=True)

        self.conv2_1 = Conv_Block_G(8 * h_channel, 4 * h_channel)
        self.conv2_2 = Conv_Block_G(4 * h_channel, 4 * h_channel, upsample=True)

        self.conv3_1 = Conv_Block_G(4 * h_channel, 2 * h_channel)
        self.conv3_2 = Conv_Block_G(2 * h_channel, 2 * h_channel, upsample=downsamplelast)

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

class PoseEnocder(nn.Module):
    """"""

    def __init__(self, dim=9, heatmap_std=0.2, in_channel=3, h_channel=32, heatmap_size=16, mode='point', downsamplelast=False):
        super(PoseEnocder, self).__init__()
        self.base = resnet18(downsamplelast=downsamplelast)
        self.conv = nn.Conv2d(256, dim, (1, 1))


    def forward(self, x):
        f = self.base(x)
        h = self.conv(f)
        return h


class PC_Encoder(nn.Module):
    def __init__(self, dim=9, downsamplelast=False):
        super().__init__()
        self.base = resnet50(downsamplelast=downsamplelast)
        self.pose_conv = Conv_Block_D(1024, dim)
        self.content_conv = Conv_Block_D(1024, 256)

    def forward(self, x):
        """
        x: image 
        """
        feat = self.base(x)
        pose = self.pose_conv(feat)
        content = self.content_conv(feat)
        return pose, content


class SCOPSP2(nn.Module):
    def __init__(self, dim=9, 
            h_channel=32, 
            item_map=dict(x='x', y='y', img='img'),
            downsamplelast=False,
            **kw):
        """
        It should be noted all params has been fixed to Jakab 2018 paper.
        Goto the original class if params and layers need to be changed.
        Images should be rescaled to 128*128
        """
        super(SCOPSP2, self).__init__()
        self.pc_encoder = PC_Encoder(dim=dim, downsamplelast=downsamplelast)
        self.generator = MyGenerator(channels=256+2, h_channel=h_channel, downsamplelast=downsamplelast)

        self.item_map = item_map

    def full_pipeline(self, x, y):
        """
        x,y: B X C X H X W (B,3,128,128)
        """
        # x info
        # content_x = self.content_encoder(x) # ([B, 256, 16, 16])
        # hm_x = self.pose_encoder(x) # B,9,16,16
        hm_x, content_x = self.pc_encoder(x)
        hm_x_sm = torch.nn.functional.softmax(hm_x, 1)
        feats_x = (hm_x_sm.unsqueeze(2) * content_x.unsqueeze(1)).sum((-1,-2), keepdim=True) # B,9,256,1,1
        feats_x = feats_x / hm_x_sm.unsqueeze(2).sum((-1,-2), keepdim=True) # normalize

        # y infon   
        # content_y = self.content_encoder(y) # ([B, 256, 16, 16])
        # hm_y = self.pose_encoder(y)
        hm_y, content_y = self.pc_encoder(y)
        hm_y_sm = torch.nn.functional.softmax(hm_y, 1)
        feats_y = (hm_y_sm.unsqueeze(2) * content_y.unsqueeze(1)).sum((-1,-2), keepdim=True) # B,9,256,1,1
        feats_y = feats_y / hm_y_sm.unsqueeze(2).sum((-1,-2), keepdim=True)
        
        B,_,Hi,Wi = content_x.shape
        device = x.device
        iy = torch.linspace(0, 1, Hi, device=device) 
        ix = torch.linspace(0, 1, Wi, device=device)
        iy = iy.reshape(1, 1, Hi, 1).repeat((B, 1, 1, Wi))
        ix = ix.reshape(1, 1, 1, Wi).repeat((B, 1, Hi, 1))

        # x2y
        code = (hm_y_sm.unsqueeze(2) * feats_x).sum(1) # B,256,16,16
        code = torch.cat((code, ix, iy), dim=1) # B,258,16,16
        recovered_y = self.generator(code) 

        # y2x
        code = (hm_x_sm.unsqueeze(2) * feats_y).sum(1) # B,256,16,16
        code = torch.cat((code, ix, iy), dim=1) # B,258,16,16
        recovered_x = self.generator(code) 

        # part_sum
        part_sum_x = hm_x_sm.sum((-1, -2))
        part_sum_y = hm_y_sm.sum((-1, -2))
        part_sum_all = part_sum_x * part_sum_y

        res = {
            'recovered_x': recovered_x,
            'recovered_y': recovered_y,
            'hm_x_sm': hm_x_sm,
            'hm_y_sm': hm_y_sm,
            'hm_x': hm_x,
            'hm_y': hm_y,
            'feats_x': feats_x,
            'feats_y': feats_y,
            'part_sum_x': part_sum_x,
            'part_sum_y': part_sum_y,
            'part_sum_all': part_sum_all,
            'content_x': content_x,
            'content_y': content_y,
        }
        return res

    def get_hm_pred(self, img):
        hm, content = self.pc_encoder(img)
        hm_sm = torch.nn.functional.softmax(hm, 1)
        res = {
            'hm_sm': hm_sm
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
        elif mode == 'get_hm_pred':
            return self.get_hm_pred(ddata['img'])
        else:
            raise ValueError('invalid mode `%s`' % mode)

    def gen_parameters(self):
        return chain(self.content_encoder.parameters(), self.generator.parameters())

