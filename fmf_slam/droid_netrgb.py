import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3
from geom.ba import BA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean


def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data

def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return .01 * eta, upmask


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)        
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, eta, upmask

        else:
            return net, delta, weight

class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""
    def __init__(self):
        super(PKT, self).__init__()
        self.channel_wise_adaptations = nn.Linear(128, 128) 
        self.spatial_wise_adaptations = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, f_s, f_t):
        # distance_loss = self.cosine_similarity_loss(f_s, f_t)
        other_loss = self.new_loss(f_s, f_t)
        return other_loss
    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        pixel_wise_euclidean_distance = torch.norm(output_net - target_net, p=2, dim=1)
        mean_pixel_wise_euclidean_distance = pixel_wise_euclidean_distance.sum()
        return 0.0001*mean_pixel_wise_euclidean_distance
    def new_loss(self, teacher_feat, student_feat):
        # print(teacher_feat.shape,student_feat.shape)#torch.Size([5, 128, 48, 64]) torch.Size([5, 128, 48, 64])
        dist_loss = self.cosine_similarity_loss(teacher_feat, student_feat)
        # kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0
        loss_dict = dict()

        student_B, student_C, student_H, student_W = student_feat.size()
        teacher_B, teacher_C, teacher_H, teacher_W = teacher_feat.size()
        assert student_B == teacher_B
        B = student_B

        # FIXME how to project student channel to teacher channel in a natural manner?
        kd_channel_loss += torch.dist(torch.mean(teacher_feat, [2, 3]),self.channel_wise_adaptations(torch.mean(student_feat, [2, 3])))*0.2 #* kd_spatial_loss_weight # 4e-3 * 6
        t_spatial_pool = torch.mean(teacher_feat, [1], keepdim=True).view(teacher_B, 1, teacher_H, teacher_W)
        s_spatial_pool = torch.mean(student_feat, [1], keepdim=True).view(student_B, 1, student_H, student_W)
        kd_spatial_loss += torch.dist(t_spatial_pool,
                            self.spatial_wise_adaptations(s_spatial_pool))*0.02 #* kd_spatial_loss_weight # 4e-3 * 6
        # print(dist_loss,kd_channel_loss,kd_spatial_loss)
        return dist_loss + kd_channel_loss + kd_spatial_loss


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = UpdateModule()
        self.correlations = PKT()
        # self.conv_layers1 = nn.Conv2d(512, 256, kernel_size=1)
        # self.conv_layers1s = nn.Conv2d(256, 128, kernel_size=1)
        # self.conv_layers2 = nn.Conv2d(512, 256, kernel_size=1)
        # self.conv_layers2s = nn.Conv2d(256, 128, kernel_size=1)

    def extract_features(self, images,depth):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])
        # depth = depth / 255.0
        # print(images.shape,depth.shape)
        # depth = depth.unsqueeze(2)
        # print(images.shape,depth.shape)
        fmaps = self.fnet([images,depth])
        # normalized_fmap1 = F.normalize(fmap1, p=2, dim=1)
        # normalized_fmap2 = F.normalize(fmap2, p=2, dim=1)

        # loss_feature = self.correlations(normalized_fmap1,normalized_fmap2)
        # fmaps = torch.cat([fmap1, fmap2], 1)
        # fmaps = self.conv_layers1(fmaps.float())
        # fmaps = self.conv_layers1s(fmaps.float())
        net = self.cnet([images,depth])
        # net = torch.cat([net1, net2], 1)
        # net = self.conv_layers2(net.float())
        net = net.unsqueeze(0)
        # net = self.conv_layers2s(net.float())
        # print(net.shape)
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps.unsqueeze(0), net, inp, 0.0


    def forward(self, Gs, images,depth, disps, intrinsics, graph=None, num_steps=12, fixedp=2):
        """ Estimates SE3 or Sim3 between pair of frames """
        # print(depth.shape)
        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        fmaps, net, inp, loss_feature = self.extract_features(images,depth)
        net, inp = net[:,ii], inp[:,ii]
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)

        ht, wd = images.shape[-2:]
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)
        
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        target = coords1.clone()

        Gs_list, disp_list, residual_list = [], [], []
        for step in range(num_steps):
            Gs = Gs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target = target.detach()

            # extract motion features
            corr = corr_fn(coords1)
            resd = target - coords1
            flow = coords1 - coords0

            motion = torch.cat([flow, resd], dim=-1)
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            net, delta, weight, eta, upmask = \
                self.update(net, inp, corr, motion, ii, jj)

            target = coords1 + delta

            for i in range(2):
                Gs, disps = BA(target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)

            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            residual = (target - coords1)

            Gs_list.append(Gs)
            disp_list.append(upsample_disp(disps, upmask))
            residual_list.append(valid_mask * residual)


        return Gs_list, disp_list, residual_list, loss_feature

