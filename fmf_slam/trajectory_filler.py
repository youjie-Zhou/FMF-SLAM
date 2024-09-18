import cv2
import torch
import lietorch

from lietorch import SE3
from collections import OrderedDict
from factor_graph import FactorGraph
from droid_net import DroidNet
import geom.projective_ops as pops
import numpy as np

class PoseTrajectoryFiller:
    """ This class is used to fill in non-keyframe poses """

    def __init__(self, net, video, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update
        self.conv_layers1s = net.conv_layers1s
        self.conv_layers2 = net.conv_layers2
        self.count = 0
        self.video = video
        self.device = device

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    # def __feature_encoder(self, image):
    #     """ features for correlation volume """
    #     return self.fnet(image)

    def __feature_encoder(self, image):
        """ features for correlation volume """
        gmap1,gmap2 = self.fnet(image)
        fmaps = torch.cat([gmap1,gmap2], 1)
        gmap = self.conv_layers1s(fmaps.float())
        return gmap#.squeeze(0)

    def __fill(self, tstamps, images, depth, intrinsics):
        """ fill operator """
        # depth = depth.unsqueeze(0)
        tt = torch.as_tensor(tstamps, device="cuda")
        images = torch.stack(images, 0)
        #print('depth1.shape',depth[1].shape)
        depth = torch.stack(depth, 0)
        #print('depth2.shape',depth[1].shape)
        intrinsics = torch.stack(intrinsics, 0)
        inputs = images[:,:,[2,1,0]].to(self.device) / 255.0
        #print('depth3.shape',depth.shape)
        ### linear pose interpolation ###
        N = self.video.counter.value
        M = len(tstamps)

        ts = self.video.tstamp[:N]
        Ps = SE3(self.video.poses[:N])

        t0 = torch.as_tensor([ts[ts<=t].shape[0] - 1 for t in tstamps])
        t1 = torch.where(t0<N-1, t0+1, t0)

        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()

        v = dP.log() / dt.unsqueeze(-1)
        w = v * (tt - ts[t0]).unsqueeze(-1)
        Gs = SE3.exp(w) * Ps[t0]

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        depth.unsqueeze(1)
        depth.unsqueeze(2)
        #print('depth4.shape',depth.shape)
        depths = depth[None,None,:]#[1, 1, 3, 384, 512]
        #print('depth5.shape',depths.shape)
        depths = np.transpose(depths,(2,0,1,3,4))
        #print('depth4.shape',depths[1].shape)
        #print('depth6.shape',depths.shape)
        depths = depths.repeat(1,1, 3, 1, 1)
        depths = torch.as_tensor(depths).type(torch.FloatTensor).to(self.device)

        inputss = [inputs, depths]
        # print(inputss[0].shape,inputss[1].shape)#torch.Size([16, 1, 3, 384, 512]) torch.Size([16, 1, 1, 384, 512])
        fmap = self.__feature_encoder(inputss)
        # print("fmap",fmap.shape)#torch.Size([16, 128, 48, 64])
        fmap = fmap.unsqueeze(1)
        # fmap = fmap.permute(1,0,2,3,4)
        # print('0',fmap.shape)
        # fmap = fmap.unsqueeze(1)
        # print('1',fmap.shape)
        self.video.counter.value += M
        self.video[N:N+M] = (tt, images[:,0], Gs.data, None, None, intrinsics / 8.0, fmap)

        graph = FactorGraph(self.video, self.update)
        graph.add_factors(t0.cuda(), torch.arange(N, N+M).cuda())
        graph.add_factors(t1.cuda(), torch.arange(N, N+M).cuda())

        for itr in range(6):
            graph.update(N, N+M, motion_only=True)
    
        Gs = SE3(self.video.poses[N:N+M].clone())
        self.video.counter.value -= M

        return [ Gs ]

    @torch.no_grad()
    def __call__(self, image_stream):
        """ fill in poses of non-keyframe images """

        # store all camera poses
        pose_list = []

        tstamps = []
        images = []
        intrinsics = []
        depths = []
        #depthsss = []
        for (tstamp, image, depth,intrinsic) in image_stream:
            tstamps.append(tstamp)
            images.append(image)
            intrinsics.append(intrinsic)
            depths.append(depth)
            #depthsss.append(depthss)
            if len(tstamps) == 16:
                pose_list += self.__fill(tstamps, images,depths, intrinsics)
                tstamps, images, intrinsics,depths = [], [], [], []

        if len(tstamps) > 0:
            pose_list += self.__fill(tstamps, images,depths, intrinsics)

        # stitch pose segments together
        return lietorch.cat(pose_list, 0)

