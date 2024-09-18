import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock
from thop import profile
import time
class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update
        self.conv_layers1s = net.conv_layers1s
        self.conv_layers2 = net.conv_layers2
        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net1, net2 = self.cnet(image)
        net = torch.cat([net1,net2], 1)
        nmap = self.conv_layers2(net.float())
        # print(nmap.shape)
        net, inp = nmap.split([128,128], dim=1)
        return net.tanh(), inp.relu()#.squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        gmap1,gmap2 = self.fnet(image)
        # print(gmap1.max(),gmap1.min(),gmap2.max(),gmap2.mean())
        fmaps = torch.cat([gmap1,gmap2], 1)
        gmap = self.conv_layers1s(fmaps.float())
        # print(gmap1.max(),gmap1.min(),gmap2.max(),gmap2.mean())
        return gmap#.squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        depth = depth#/255.0
        depth = torch.as_tensor(depth).type(torch.FloatTensor).to(self.device)

        depthss = depth.unsqueeze(0).unsqueeze(0).unsqueeze(0)#.permute(0,1,4,2,3)
        depthss = depthss.expand(-1, -1, 3, -1, -1)
        # print('1',inputs.max(),inputs.mean(),depthss.max(),depthss.mean())(2.5877,0.2732,3.7893,0.9243)
        inputss = [inputs, depthss]
        # extract features
        # print(inputss[0].shape,inputss[1].shape)#torch.Size([1, 1, 3, 384, 512]) torch.Size([1, 1, 3, 384, 512])

        # model = self.__feature_encoder()  
        # start_time = time.perf_counter()  # 记录开始时间
        gmap = self.__feature_encoder(inputss)
        # end_time = time.perf_counter()  # 记录结束时间
        # elapsed_time = end_time - start_time  # 计算运行时间   
        # print(f"total: {elapsed_time} 秒")    
        # print('gmap',gmap.shape)#torch.Size([1, 128, 48, 64])
        # fmaps = torch.cat([gmap1,gmap2], 1)
        # gmap = self.conv_layers1s(fmaps.float())
        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            inputss = [inputs[:, [0]], depthss[:, [0]]]
            net, inp = self.__context_encoder(inputss)
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(tstamp, image[0], Id, 1.0, None, intrinsics / 8.0, gmap, net[0,0], inp[0,0])

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                inputss = [inputs[:, [0]], depthss[:, [0]]]
                net, inp = self.__context_encoder(inputss)
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(tstamp, image[0], None, None, None, intrinsics / 8.0, gmap, net[0], inp[0])

            else:
                self.count += 1




# class MotionFilter:
#     """ This class is used to filter incoming frames and extract features """

#     def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
#         # split net modules
#         self.cnet = net.cnet
#         self.fnet = net.fnet
#         self.update = net.update

#         self.video = video
#         self.thresh = thresh
#         self.device = device

#         self.count = 0

#         # mean, std for image normalization
#         self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
#         self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
#     @torch.cuda.amp.autocast(enabled=True)
#     def __context_encoder(self, image):
#         """ context features """
#         x = self.cnet(image)
#         net, inp = self.cnet(image).split([128,128], dim=2)
#         return net.tanh().squeeze(0), inp.relu().squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     def __feature_encoder(self, image):
#         """ features for correlation volume """
#         return self.fnet(image).squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     @torch.no_grad()
#     def track(self, tstamp, image, depth=None, intrinsics=None):
#         """ main update operation - run on every frame in video """

#         Id = lietorch.SE3.Identity(1,).data.squeeze()
#         ht = image.shape[-2] // 8
#         wd = image.shape[-1] // 8

#         # normalize images
#         inputs = image[None, None, [2,1,0]].to(self.device) / 255.0
#         inputs = inputs.sub_(self.MEAN).div_(self.STDV)

#         # extract features
#         gmap = self.__feature_encoder(inputs)

#         ### always add first frame to the depth video ###
#         if self.video.counter.value == 0:
#             net, inp = self.__context_encoder(inputs)
#             self.net, self.inp, self.fmap = net, inp, gmap
#             self.video.append(tstamp, image, Id, 1.0, intrinsics / 8.0, gmap[0], net[0], inp[0])

#         ### only add new frame if there is enough motion ###
#         else:                
#             # index correlation volume
#             coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
#             corr = CorrBlock(self.fmap[None], gmap[None])(coords0)

#             # approximate flow magnitude using 1 update iteration
#             _, delta, weight = self.update(self.net[None], self.inp[None], corr)

#             # check motion magnitue / add new frame to video
#             if delta.norm(dim=-1).mean().item() > self.thresh:
#                 self.count = 0
#                 net, inp = self.__context_encoder(inputs)
#                 self.net, self.inp, self.fmap = net, inp, gmap
#                 self.video.append(tstamp, image, None, None, intrinsics / 8.0, gmap[0], net[0], inp[0])

#             else:
#                 self.count += 1

