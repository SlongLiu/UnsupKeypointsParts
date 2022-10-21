from loss.arcloss import ArcMarginProduct
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .baseloss import BaseLoss
from utils.utils import AverageMeter, get_lm_map
from loss.crossentropyweightedloss import cross_entropy_with_weights

class FeatDistLoss(BaseLoss):
    def __init__(self, item_map=dict(feat='feat', fix_feat='fix_feat'), **kw):
        super(FeatDistLoss, self).__init__(item_map)

    def run(self, feat, fix_feat):
        loss = (feat - fix_feat)**2
        return loss.mean()

class FeatDistCosLoss(BaseLoss):
    def __init__(self, item_map=dict(feat='feat', fix_feat='fix_feat', weight='weight'), **kw):
        super(FeatDistCosLoss, self).__init__(item_map)

    def run(self, feat, fix_feat, weight=None):
        weight = weight / weight.sum(-1, keepdim=True) # B,N
        feat = feat.squeeze(-1).squeeze(-1)
        fix_feat = fix_feat.squeeze(-1).squeeze(-1)
        # print('feat.shape:', feat.shape)
        loss = (feat*fix_feat).sum((-1)) / torch.norm(feat, dim=-1) / torch.norm(fix_feat, dim=-1)
        loss = loss * weight
        return (1-loss).mean()


class FeatOrthLoss(BaseLoss):
    def __init__(self, item_map=dict(feat='feat'), **kw):
        super(FeatOrthLoss, self).__init__(item_map)

    def run(self, feat):
        """
        loss = ||W*(W^T)-I||
        Args:
            feat (torch.Tensor): B,N,M
        """
        if isinstance(feat, (list, tuple)):
            feat = torch.cat(feat, dim=2)
        # print('feat.shape:', feat.shape)
        shapes = feat.shape
        if feat.dim() > 3:
            feat = feat.reshape(shapes[0], shapes[1], -1)
        feat = feat / torch.norm(feat, p=2, dim=2, keepdim=True)
        corr = feat.bmm(feat.permute(0,2,1)) # B,N,N
        # print('feat:', feat.shape)
        # print('corr:', corr)
        # raise ValueError
        I = torch.eye(corr.size(1)).unsqueeze(0).to(corr.device)
        loss = torch.norm(corr - I)
        return loss


class FeatDistFixLoss(BaseLoss):
    def __init__(self, fix_feat_path, item_map=dict(feat='feat'), **kw):
        super(FeatDistFixLoss, self).__init__(item_map)
        self.fix_feat = torch.Tensor(np.load(fix_feat_path)).unsqueeze(0)

    def run(self, feat):
        fix_feat = self.fix_feat.to(feat.device)
        loss = (feat - fix_feat)**2
        return loss.mean()


class FeatDistPercepLoss(BaseLoss):
    def __init__(self, fix_feat_path, refresh_iterval=100, item_map=dict(feat='feat'), **kw):
        super(FeatDistPercepLoss, self).__init__(item_map)
        self.fix_feat = torch.Tensor(np.load(fix_feat_path)).unsqueeze(0)
        # print(self.fix_feat.shape)
        # raise ValueError
        # 960 -> (64, 128, 256, 512)
        self.fix_feat_list = (
            self.fix_feat[:, :,  :64],
            self.fix_feat[:, :, 64:64+128],
            self.fix_feat[:, :, 64+128:64+128+256],
            self.fix_feat[:, :, 64+128+256:]
        )

        self.alpha = [1, 1, 1, 1]
        self.cnt = 0

        # moving average
        self.mov_a = [AverageMeter() for i in range(4)]
        self.refresh_iterval = refresh_iterval

    def run(self, featlist):
        res = {}
        for idx, (feat, fix_feat) in enumerate(zip(featlist, self.fix_feat_list)):
            # print('fix_feat.shape:', fix_feat.shape)
            fix_feat = fix_feat.to(feat.device)
            loss_i = ((feat - fix_feat)**2).mean() * self.alpha[idx]
            self.mov_a[idx].update(loss_i.item(), 1)

            res.update({str(idx): loss_i })

        # update the parameter
        # self.cnt = self.cnt + 1
        if (self.mov_a[-1].cnt+1) % self.refresh_iterval == 0:
            for i in range(4):
                self.alpha[i] = self.alpha[i] / self.mov_a[i].avg
                # self.alpha[i].reset()
                self.mov_a[i].reset()

        return res


# class TwoFeatDistPercepLoss(BaseLoss):
#     def __init__(self, refresh_iterval=10, item_map=dict(featlist='feat', fix_featlist='fix_feat'), **kw):
#         super(TwoFeatDistPercepLoss, self).__init__(item_map)
#         self.alpha = [0, 0, 0, 1]
#         self.cnt = 0

#         # moving average
#         self.mov_a = [AverageMeter() for i in range(4)]
#         self.refresh_iterval = refresh_iterval

#     def run(self, featlist, fix_featlist):
        
#         fix_featlist = (
#             fix_featlist[:, :,  :64],
#             fix_featlist[:, :, 64:64+128],
#             fix_featlist[:, :, 64+128:64+128+256],
#             fix_featlist[:, :, 64+128+256:]
#         )

#         res = {}
#         for idx, (feat, fix_feat) in enumerate(zip(featlist, fix_featlist)):
#             # print('fix_feat.shape:', fix_feat.shape)
#             if self.alpha[idx] == 0:
#                 continue
#             fix_feat = fix_feat.to(feat.device)
#             # print('fix_feat:', fix_feat)
#             # print('feat:', feat)
#             loss_i = ((feat - fix_feat)**2).mean() * self.alpha[idx]
#             self.mov_a[idx].update(loss_i.item(), 1)

#             res.update({str(idx): loss_i})

#         # print([self.mov_a[idx].cnt for idx in range(4)])

#         # raise ValueError

#         # update the parameter
#         # self.cnt = self.cnt + 1
#         # print('self.cnt:', self.cnt)
#         # if (self.mov_a[-1].cnt+1) % self.refresh_iterval == 0:
#         #     print(self.mov_a[-1].cnt+1, 'self.alpha:', self.alpha)
#         #     for i in range(4):
#         #         if self.alpha[i] == 0:
#         #             continue
#         #         self.alpha[i] = self.alpha[i] / self.mov_a[i].avg
#         #         self.mov_a[i].reset()
            
#         return res

class TwoFeatDistPercepLoss(BaseLoss):
    def __init__(self, refresh_iterval=10, item_map=dict(featlist='feat', fix_featlist='fix_feat'), **kw):
        super(TwoFeatDistPercepLoss, self).__init__(item_map)
        self.metric = nn.MSELoss()


    def run(self, feat, fix_feat):
        # return ((feat - fix_feat)**2).mean()
        return self.metric(feat, fix_feat)

# class TwoFeatDistRelativeLoss(BaseLoss):
#     def __init__(self, refresh_iterval=10, item_map=dict(featlist='feat', fix_featlist='fix_feat'), **kw):
#         super(TwoFeatDistRelativeLoss, self).__init__(item_map)
#         self.metric = nn.CrossEntropyLoss()


#     def run(self, feat, fix_feat):
#         # feat, fix_feat: B,10,960
#         fix_feat = fix_feat.repeat((feat.shape[0], 1, 1))
#         corr = feat.bmm(fix_feat.permute(0, 2, 1)) # B,10,10
#         corr = corr.reshape(-1, corr.shape[-1])
#         gt = torch.arange(feat.shape[1]).repeat(feat.shape[0]).to(corr.device)
#         return self.metric(corr, gt)

class TwoFeatContrastiveLoss(BaseLoss):
    def __init__(self, therhold=1, item_map=dict(featlist='feat', fix_featlist='fix_feat'), **kw):
        super(TwoFeatContrastiveLoss, self).__init__(item_map)
        self.metric = nn.CrossEntropyLoss()
        self.therhold = therhold

    def run(self, feat, fix_feat):
        # feat, fix_feat: B,10,960
        fix_feat = feat.mean(0, keepdim=True)
        dist_mat = ((feat.unsqueeze(2) - fix_feat.unsqueeze(1))**2).sum(-1) # B,10,10
        dist_mat = dist_mat / dist_mat.sum(-1, keepdim=True) * dist_mat.shape[-1]
        # print('dist_mat:', dist_mat[0])
        ni = dist_mat.shape[1]
        ident = torch.eye(ni, ni).repeat((dist_mat.shape[0], 1,1)).to(dist_mat.device)
        # dist_mat_sm = torch.softmax(-dist_mat, -1)
        # print('dist_mat_sm.shape:', dist_mat_sm.shape)
        # print('dist_mat_sm:', dist_mat_sm[0])
        # raise ValueError
        # loss = ident * dist_mat_sm + (1 - ident) * torch.max(torch.zeros_like(dist_mat_sm).to(dist_mat_sm.device), self.therhold - dist_mat_sm)
        loss = ident * dist_mat + (1 - ident) * torch.max(torch.zeros_like(dist_mat).to(dist_mat.device), self.therhold - dist_mat)
        return loss.mean()


class HardMineTripletLoss(BaseLoss):
    def __init__(self, therhold=1, item_map=dict(featlist='feat'), **kw) -> None:
        super(HardMineTripletLoss, self).__init__(item_map)
        self.therhold = therhold

    def run(self, feat):
        B,N,L = feat.shape
        l_dist = ((feat.unsqueeze(2) - feat.unsqueeze(1))**2).sum(-1) # B,10,10
        v_dist = ((feat.unsqueeze(1) - feat.unsqueeze(0))**2).sum(-1) # B,B,10
        ident = torch.eye(N, N).repeat((B, 1,1)).to(l_dist.device) * 1e12
        n_dist = (l_dist + ident).min(-1)[0]
        p_dist = v_dist.max(1)[0] # B, 10
        # loss = torch.nn.functional.relu(p_dist - n_dist + self.therhold)
        loss = torch.abs(p_dist - n_dist + self.therhold)
        return loss.mean()


class HardMineCMTripletLoss(BaseLoss):
    def __init__(self, therhold=0.2, item_map=dict(feat='feat'), **kw) -> None:
        super(HardMineCMTripletLoss, self).__init__(item_map)
        self.therhold = therhold

    def run(self, feat):
        B,N,L = feat.shape
        feat_norm = torch.norm(feat, dim=-1) # B,10
        l_dist = (feat.unsqueeze(2) * feat.unsqueeze(1)).sum(-1) / feat_norm.unsqueeze(2) / feat_norm.unsqueeze(1)
        v_dist = (feat.unsqueeze(1) * feat.unsqueeze(0)).sum(-1) / feat_norm.unsqueeze(1) / feat_norm.unsqueeze(0)
        ident = torch.eye(N, N).repeat((B, 1,1)).to(l_dist.device) * 1
        n_dist = (l_dist + ident).min(-1)[0]
        p_dist = v_dist.max(1)[0] # B, 10
        # loss = torch.nn.functional.relu(p_dist - n_dist + self.therhold)
        loss = torch.abs(p_dist - n_dist + self.therhold)
        return loss.mean()

def cosine_similarity(x, y):
    """[summary]

    Args:
        x (tensor): B,N,L
        y (tensor): B,N,L

    Returns:
        [type]: [description]
    """
    return (x*y).sum(-1) / torch.norm(x, dim=-1) / torch.norm(y, dim=-1)

def euclid_distance(x, y):
    return ((x-y)**2).sum(-1)



class LFCenterLoss(BaseLoss):
    def __init__(self, metric='l2', item_map=dict(feat='feat'), **kw) -> None:
        super(LFCenterLoss, self).__init__(item_map)
        assert metric in ['l2', 'cos']
        self.metric = metric
        self.f = euclid_distance if metric == 'l2' else cosine_similarity
        
    def run(self, feat):
        B,N,L = feat.shape
        # print(feat.shape)
        feat_mean = feat.mean(0, keepdim=True)
        dist = self.f(feat, feat_mean)
        if self.metric == 'cos':
            dist = (1 - dist)
        return dist.mean()


class LFLineCenterLoss(BaseLoss):
    def __init__(self, metric='l2', item_map=dict(feat='feat'), **kw) -> None:
        super(LFLineCenterLoss, self).__init__(item_map)
        assert metric in ['l2', 'cos']
        self.metric = metric
        self.f = euclid_distance if metric == 'l2' else cosine_similarity
        
    def run(self, feat):
        B,N,L = feat.shape
        # print(feat.shape)
        feat_mean = feat.mean(1, keepdim=True)
        dist = self.f(feat, feat_mean)
        if self.metric == 'cos':
            dist = (1 - dist)
        return dist.mean()


#########################################
# color feature loss
#########################################


class ColorFeatLoss(BaseLoss):
    def __init__(self, item_map=dict(pred='pred', visb='visb'), **kw):
        super(ColorFeatLoss, self).__init__(item_map)
        # self.metric = nn.CrossEntropyLoss()
        lm_map = get_lm_map()
        gt = np.zeros(68)
        for i in range(68):
            gt[i] = lm_map[i]
        self.gt = torch.Tensor(gt).type(torch.LongTensor)
        # self.gt = nn.Parameter(gt) # B,9

        # self.gt.requires_grad = False

    def run(self, pred, visb):
        """
        Args:
            pred ([type]): B,68,9
            gt ([type]): 68
            visb : B,68
        """
        gt = self.gt.to(pred.device).repeat((pred.shape[0]))
        pred = pred.view(-1, pred.shape[-1])
        loss = cross_entropy_with_weights(pred, gt, visb.view(-1))
        
        return loss

class InverseColorLoss(BaseLoss):
    def __init__(self, item_map=dict(hm='hm', c_idx_map='c_idx_map'), therhold=1.0, **kw):
        super(InverseColorLoss, self).__init__(item_map)
        self.therhold = therhold

    def run(self, hm, c_idx_map):
        """[summary]

        Args:
            hm (B,68,16,16): [description]
            cm (B,K,16,16): [description]
        """
        hm = hm.sum(1, keepdim=True) # B,1,16,16
        vm = (hm * c_idx_map).sum((2, 3)) # B,K
        loss = F.relu(self.therhold - vm)
        return loss.mean()


############################################################
# SCOPS

class GroupConcenLoss(BaseLoss):
    def __init__(self, item_map=dict(feat='feat'), **kw):
        super(GroupConcenLoss, self).__init__(item_map)

    def run(self, feat):
        feat = feat.squeeze(-1).squeeze(-1) # B,9,256
        mean_feat = feat.mean(0, keepdim=True) # 1,9,256
        loss = ((feat - mean_feat)**2).sum(-1).mean(0)[1:].mean()
        return loss


class ArcFaceLoss(BaseLoss):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, item_map=dict(feat='feat'), skip_bg=True, **kw) -> None:
        super(ArcFaceLoss, self).__init__(item_map)
        self.metric = ArcMarginProduct(in_features=in_features, out_features=out_features, s=s, m=m, easy_margin=easy_margin)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.skip_bg = skip_bg

    def run(self, feat):
        # print('Get feat shape:', feat.shape)
        feat = feat.squeeze(-1).squeeze(-1) # B,9,256
        B,N,L = feat.shape
        device = feat.device
        loss = 0
        for i in range(N):
            if i==0 and self.skip_bg:
                continue
            feat_i = feat[:,i,:] # B,256
            label = (torch.ones(B)*i).to(device) # B
            # print('feat_i.shape:', feat_i.shape)
            loss += self.criterion(self.metric(feat_i, label), label.long())
            # print('loss:', loss)
        return loss / N


    