
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp

from .augmentation import RGBDAugmentor
from .rgbd_utils import *
from scipy.ndimage import gaussian_filter
class RGBDDataset(data.Dataset):
    def __init__(self, name, datapath, n_frames=4, crop_size=[304,400], fmin=8.0, fmax=75.0, do_aug=True):#384,512  400 304
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = datapath
        self.name = name

        self.n_frames = n_frames
        self.fmin = fmin # exclude very easy examples
        self.fmax = fmax # exclude very hard examples
        
        if do_aug:
            self.aug = RGBDAugmentor(crop_size=crop_size)

        # building dataset is expensive, cache so only needs to be performed once
        cur_path = osp.dirname(osp.abspath(__file__))
        if not os.path.isdir(osp.join(cur_path, 'cache')):
            os.mkdir(osp.join(cur_path, 'cache'))
        
        cache_path = osp.join(cur_path, 'cache', '{}.pickle'.format(self.name))

        if osp.isfile(cache_path):
            scene_info = pickle.load(open(cache_path, 'rb'))[0]
        else:
            scene_info = self._build_dataset()
            with open(cache_path, 'wb') as cachefile:
                pickle.dump((scene_info,), cachefile)

        self.scene_info = scene_info
        self._build_dataset_index()
                
    def _build_dataset_index(self):
        self.dataset_index = []
        for scene in self.scene_info:
            if not self.__class__.is_test_scene(scene):
                graph = self.scene_info[scene]['graph']
                for i in graph:
                    if len(graph[i][0]) > self.n_frames:
                        self.dataset_index.append((scene, i))
            else:
                print("Reserving {} for validation".format(scene))

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)
        # return cv.imread(depth_file)
    def preprocess_depths(depth):
            # 将深度值小于100的像素点设为0
            # print('original',depth.max(),depth.min(),np.median(depth))
            colored_depth_images = []
            batchsize,_,_ = depth.shape
            depth_image = depth 
            depth_image[depth_image>20.0] = 0 
            # print('1',depth.shape,depth_image.shape) 
            # depth_image = 1.0 / depth_image
            depth_image_normalized =  depth_image/depth_image.max()
            # print('111',depth_image.max())
            for i in range(batchsize):
                colored_depth_image = cv2.applyColorMap((depth_image_normalized[i]*250.0).astype(np.uint8), cv2.COLORMAP_JET)
                # print('1',colored_depth_image.shape)
                colored_depth_images.append(colored_depth_image)
            colored_depth_images = np.stack(colored_depth_images, axis=0)
            # print('2',colored_depth_images.shape)
            return colored_depth_images
    # def preprocess_depths(depth):
    #         # 将深度值小于100的像素点设为0
    #         # print('original',depth.max(),depth.min(),np.median(depth))
            
    #         inv_depth = 1.0/(depth + 0.0001)
    #         depth[depth>30.0] = 0
    #         depth = depth/depth.max()
    #         # print(inv_depth.shape,depth.shape,depth.max(),depth.min(),depth.mean())
    #         result = np.stack((inv_depth, depth), axis=-1)
    #         # result = torch.cat((inv_depth,depth ), dim=-1)
    #         return result    
    
    # def preprocess_depths(depth):
    #         # 将深度值小于100的像素点设为0
    #         # print('original',depth.max(),depth.min(),np.median(depth))
    #         colored_depth_images = []
    #         batchsize,_,_ = depth.shape
    #         depth_image = depth *10
    #         depth_image[depth_image>250.0] = 0 
    #         # print('1',depth.shape,depth_image.shape) 
    #         # depth_image = 1.0 / depth_image
    #         depth_image_normalized =  depth_image #depth_image/depth_image.max()
    #         for i in range(batchsize):
    #             colored_depth_image = cv2.applyColorMap((depth_image_normalized[i]).astype(np.uint8), cv2.COLORMAP_JET)
    #             # print('1',colored_depth_image.shape)
    #             colored_depth_images.append(colored_depth_image)
    #         colored_depth_images = np.stack(colored_depth_images, axis=0)
    #         # print('2',colored_depth_images.shape)
    #         return colored_depth_images
    # 获取深度图的形状
    def preprocess_depth(depth_data,images):
        colored_depth_images = []
        # print(depth_data.shape)
        batchsize,height, width = depth_data.shape
        # 计算每块的大小
        block_height = height // 160
        block_width = width // 160
        # 初始化最终的逆深度图
        final_inverse_depth = np.zeros_like(depth_data)
        # 循环遍历每块
        for i in range(160):
            for j in range(160):
                # 计算当前块的起始和结束位置
                start_row = i * block_height
                end_row = (i + 1) * block_height
                start_col = j * block_width
                end_col = (j + 1) * block_width
                # 提取当前块的深度信息
                block_depth = depth_data[:, start_row:end_row, start_col:end_col]
                # print(block_depth.shape)
                # 计算当前块的逆深度
                if np.max(block_depth) > 600 and np.min(block_depth) > 600:
                # 设置当前块的逆深度为 1.0
                    block_inverse_depth = 1.0
                elif np.min(block_depth) /np.max(block_depth) > 0.95:
                    # 设置当前块的逆深度为 1.0
                    block_inverse_depth = 1.0
                else:
                    # 计算当前块的逆深度
                    block_inverse_depth = block_depth
                    block_inverse_depth = block_inverse_depth - np.min(block_inverse_depth)/ np.max(block_inverse_depth) - np.min(block_inverse_depth)
                # 将当前块的逆深度放入最终逆深度图中
                final_inverse_depth[:,start_row:end_row, start_col:end_col] = block_inverse_depth
        final_inverse_depth = gaussian_filter(final_inverse_depth, sigma=1)
        for i in range(batchsize):
            # 转换为灰度图像
            gray_image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            gray_image = np.uint8(gray_image)
            # 对灰度图像应用Canny边缘检测
            edges = cv2.Canny(gray_image, threshold1=150, threshold2=255)
            edge_depth_image = cv2.applyColorMap((final_inverse_depth[i] * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
            edge_depth_image = cv2.cvtColor(edge_depth_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            edge_depth_image[edge_depth_image < 120] = 0
            edge_depth_image[edge_depth_image >= 120] =255
            edge_depth_image = edge_depth_image + edges
            _, edge_depth_binary = cv2.threshold((255-edge_depth_image ).astype(np.uint8),150, 255, cv2.THRESH_BINARY)
            dist_transform = cv2.distanceTransform(edge_depth_binary, cv2.DIST_L2, 3)
            cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
            colored_depth_images.append(dist_transform)
            # print(dist_transform.shape)
        colored_depth_images = np.stack(colored_depth_images, axis=0)
        return colored_depth_images
    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f
        
        disps = np.stack(list(map(read_disp, depths)), 0)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        # uncomment for nice visualization
        # import matplotlib.pyplot as plt
        # plt.imshow(d)
        # plt.show()

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph

    def __getitem__(self, index):
        """ return training video """

        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]

        frame_graph = self.scene_info[scene_id]['graph']
        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']

        inds = [ ix ]
        while len(inds) < self.n_frames:
            # get other frames within flow threshold
            k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
            frames = frame_graph[ix][0][k]

            # prefer frames forward in time
            if np.count_nonzero(frames[frames > ix]):
                ix = np.random.choice(frames[frames > ix])
            
            elif np.count_nonzero(frames):
                ix = np.random.choice(frames)

            inds += [ ix ]

        images, depths, poses, intrinsics = [], [], [], []
        for i in inds:
            images.append(self.__class__.image_read(images_list[i]))
            depths.append(self.__class__.depth_read(depths_list[i]))
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)

        # color= self.__class__.preprocess_depths(depths)
        # depthss = np.concatenate((color, edge), axis=3)
        # depthss = depthss.float()
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)
        # print(images.shape,depthss.shape)
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        valid_mask = (depths >0.1)
        depths1 = np.where(valid_mask, depths, 0.1)
        disps = torch.from_numpy(1.0 / depths1)

        valid_masks = (depths >0.4)
        depths2 = np.where(valid_masks, depths, 0.4)
        # disps = torch.from_numpy(1.0 / depths)
        # depths[depths<0.4] = 0.4
        # print(depths.min())
        dispsss = torch.from_numpy(1.0 / depths2)
        depthss = dispsss.unsqueeze(1).repeat(1,3,1,1)
        # print(disps.shape,depthss.shape)
        # print(depths.min(),disps.mean())
        # depthss = self.__class__.preprocess_depths(depths)
        # depthss = torch.from_numpy(depths/depths.max())
        # depthss = disps.unsqueeze(2)
        # # print(depthss.shape)
        # depthss = depthss.expand(-1,-1,3,-1,-1)

        # # depthss = torch.cat((depthss.unsqueeze(1), disps.unsqueeze(1)), dim=1)
        # # depthss = depths[depths<100.0]
        # depthss = torch.from_numpy(depthss).float()
        # print(depthss.shape,images.shape,depthss.mean(),images.mean())
        # depthss = depthss.permute(0, 3, 1, 2)
        # depthss = depthss.permute(0,3,1,2)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)
        # print('original',disps.shape,depthss.shape)
        if self.aug is not None:
            images, poses, disps, intrinsics,depthss = \
                self.aug(images, poses, disps, intrinsics,depthss)
        # depthss = disps
        # print('update',disps.shape,depthss.shape)
        # scale scene
        if len(disps[disps>0.01]) > 0:
            s = disps[disps>0.01].mean()
            disps = disps / s
            poses[...,:3] *= s
            
        return images, poses, disps, intrinsics,depthss

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self
