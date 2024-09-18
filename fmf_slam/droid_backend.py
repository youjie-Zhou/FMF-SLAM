import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class DroidBackend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update

        # global optimization window
        self.t0 = 0
        self.t1 = 0


        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms
        
    @torch.no_grad()
    def __call__(self, steps=12):
        """ main update """

        t = self.video.counter.value
        if not self.video.stereo:
             self.video.normalize()



        t = self.video.counter.value

        # Gs = SE3(self.video.poses.clone()).inv().data.cpu().numpy()
        # Gs[:, 0] = self.video.gps[:, 0].cpu().numpy()/1.3148  # 经度
        # Gs[:, 2] = self.video.gps[:, 1].cpu().numpy()/1.3148  # 纬度
        # Gs_tensor = torch.tensor(Gs, dtype=torch.float32, device='cuda')
        # Gs_se3 = SE3(Gs_tensor)
        # self.video.poses = Gs_se3.inv().data

        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=16*t)

        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta)

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        self.video.dirty[:t] = True
