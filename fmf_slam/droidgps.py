import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process
from lietorch import SE3

class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self,gps, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(gps,tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            # self.backend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """



        del self.frontend

        torch.cuda.empty_cache()

        # Gs = SE3(self.video.poses.clone()).inv().data.cpu().numpy()
        # Gs[:, 0] = self.video.gps[:, 0].cpu().numpy()/1.3148  # 经度
        # Gs[:, 2] = self.video.gps[:, 1].cpu().numpy()/1.3148  # 纬度

        # Gs_tensor = torch.tensor(Gs, dtype=torch.float32, device='cuda')
        # Gs_se3 = SE3(Gs_tensor)
        # self.video.poses = Gs_se3.inv().data
        # t = self.video.counter.value
        # with open('poses1_xy.txt', 'w') as file:
        #     # 遍历self.video.poses的每一行
        #     for i in range(len(Gs)):
        #         # 提取第1个和第3个数字（即x和y坐标）
        #         x = self.video.poses[i, 0].item()
        #         y = self.video.poses[i, 2].item()
        #         # 将x和y坐标写入文件，每对坐标占一行，坐标之间用空格分隔
        #         file.write(f'{x} {y}\n')

        # with open('gps_xy.txt', 'w') as file:
        #     # 遍历self.video.poses的每一行
        #     for i in range(len(self.video.gps)):
        #         # 提取第1个和第3个数字（即x和y坐标）
        #         x = self.video.gps[i, 0].item()
        #         y = self.video.gps[i, 1].item()
        #         # 将x和y坐标写入文件，每对坐标占一行，坐标之间用空格分隔
        #         file.write(f'{x} {y}\n')

        # 更新Gs中的经纬度信息



        # with open('poses2_xy.txt', 'w') as file:
        #     # 遍历self.video.poses的每一行
        #     for i in range(len(Gs)):
        #         # 提取第1个和第3个数字（即x和y坐标）
        #         x = self.video.poses[i, 0].item()
        #         y = self.video.poses[i, 2].item()
        #         # 将x和y坐标写入文件，每对坐标占一行，坐标之间用空格分隔
        #         file.write(f'{x} {y}\n')

        # updated_poses = SE3(torch.tensor(Gs, dtype=torch.float32, device='cuda'))
        # self.video.poses = updated_poses.data
        # 打开一个文件用于写入，如果文件不存在将会被创建

        
        # print("Longitude (self.poses[:, 0]):\n", Gs[0:30,0])
        # print("Longitude (self.gps[:, 0]):\n", self.video.gps[:30, 0].cpu().numpy())
        # print("Latitude (self.poses[:, 2]):\n", Gs[0:30,2])
        # print("Latitude (self.gps[:, 1]):\n", self.video.gps[:30, 1].cpu().numpy())
        # self.video.poses[:, 0] = -self.video.gps[:, 0]  # 更新经度
        # self.video.poses[:, 2] = -self.video.gps[:, 1]  # 更新纬度
        # with self.video.get_lock():
        # self.video.poses[:100, 1] = 25
        # self.video.poses[:100, 0] = 0
        # self.video.poses[:100, 2] = 0
        # ts1 = self.video.counter.value



        Gs = SE3(self.video.poses.clone()).inv().data.cpu().numpy()
        Gs[:, 0] = self.video.gps[:, 0].cpu().numpy()#/1.3148  # 经度
        Gs[:, 2] = self.video.gps[:, 1].cpu().numpy()#/1.3148  # 纬度
        Gs_tensor = torch.tensor(Gs, dtype=torch.float32, device='cuda')
        Gs_se3 = SE3(Gs_tensor).inv().data.cuda()
        self.video.poses[:,0] = Gs_se3[:,0]
        self.video.poses[:,1] = Gs_se3[:,1]
        self.video.poses[:,2] = Gs_se3[:,2]
        self.video.poses[:,3] = Gs_se3[:,3]
        self.video.poses[:,4] = Gs_se3[:,4]
        self.video.poses[:,5] = Gs_se3[:,5]
        self.video.poses[:,6] = Gs_se3[:,6]

        # with open('poses2_xy.txt', 'w') as file:
        #     # 遍历self.video.poses的每一行
        #     for i in range(len(Gs)):
        #         # 提取第1个和第3个数字（即x和y坐标）
        #         x = self.video.poses[i, 0].item()
        #         y = self.video.poses[i, 2].item()
        #         # 将x和y坐标写入文件，每对坐标占一行，坐标之间用空格分隔
        #         file.write(f'{x} {y}\n')
        

        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()

