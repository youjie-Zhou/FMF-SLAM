import sys
sys.path.append('droid_slam')
import PIL
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import csv
from torch.multiprocessing import Process
from droid import Droid
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from skimage import exposure,io
# from data_readers.augmentation import RGBDAugmentor
# from data_readers.rgbd_utils import *
import random

def load_tum_data(dataset_path, associate_file):
    # Load associate file and extract image paths and timestamps
    with open(os.path.join(dataset_path, associate_file), 'r') as f:
        data = f.readlines()
        rgb_paths = [os.path.join(dataset_path, line.split()[1]) for line in data]
        depth_paths = [os.path.join(dataset_path, line.split()[3]) for line in data]
        timestamps = [float(line.split()[0]) for line in data]
    # print(len(rgb_paths))
    return rgb_paths, depth_paths, timestamps
# def load_tum_data(dataset_path, associate_file1, associate_file2):
#     # Load associate file1 and extract image paths, timestamps, and GPS coordinates
#     with open(os.path.join(dataset_path, associate_file1), 'r') as f:
#         data = f.readlines()
#         rgb_paths = [os.path.join(dataset_path, line.split()[1]) for line in data]
#         timestamps = [float(line.split()[0]) for line in data]
#         # gps_coords = [(float(line.split()[4].split('-')[0]), float(line.split()[4].split('-')[1])) for line in data]
#         # gps_coords = [(float(line.split()[3].split('-')[0]), float(line.split()[3].split('-')[1])) for line in data]
#         gps_coords = [(float(line.split()[3].replace('GPS/', '').split('-')[0]), float(line.split()[3].replace('GPS/', '').split('-')[1])) for line in data]
#     first_row_x, first_row_y = gps_coords[0]
#     adjusted_gps_coords = [(x - first_row_x, y - first_row_y) for x, y in gps_coords]
#     # Load associate file2 and extract depth paths
#     with open(os.path.join(dataset_path, associate_file2), 'r') as f:
#         data = f.readlines()
#         depth_paths = [os.path.join(dataset_path, line.split()[1]) for line in data]

#     return rgb_paths, depth_paths, timestamps, adjusted_gps_coords
def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, depthdir, calib, stride):
    """ image generator """
    associate_file = 'associate.txt'
    # associate_file2 = 'depth_gps.txt'
    # calib = np.loadtxt(calib, delimiter=" ")
    # fx, fy, cx, cy = calib[:4]
    K = np.eye(3)
    # K[0,0]= fx = 517.3
    # K[0,2]= cx = 318.6
    # K[1,1]= fy = 516.5
    # K[1,2]= cy = 255.3
    # K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    # d_l = np.array([0.262383, -0.953104, -0.005358, 0.002628, 1.163314])

    # K[0,0]= fx = 520.9
    # K[0,2]= cx = 325.1
    # K[1,1]= fy = 521.0
    # K[1,2]= cy = 249.7
    # K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    # d_l = np.array([0.231222, -0.784899, -0.003257, -0.000105, 0.917205])

    # K[0,0]= fx = 535.4
    # K[0,2]= cx = 320.1
    # K[1,1]= fy = 539.2
    # K[1,2]= cy = 247.6
    # K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    # d_l = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    K[0,0]= fx = 600.09
    K[0,2]= cx = 329.82
    K[1,1]= fy = 600.66
    K[1,2]= cy = 248.36
    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([0.16877, -0.507345, -0.000536, 0.000172, 0.471516])

    # K[0,0]= fx = 462.117
    # K[0,2]= cx = 310.898
    # K[1,1]= fy = 461.885
    # K[1,2]= cy = 248.93
    # K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    # d_l = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    images_list,depth_list, timestamps = load_tum_data(imagedir,associate_file)
    # images_list = images_list[::2]
    # depth_list = depth_list[::2]
    # timestamps = timestamps[::2]
    for t, imfile in enumerate(images_list):
        image = cv2.imread(imfile)
        # depth_path = os.path.join(datapath, 'depth')
        depth = cv2.imread(depth_list[t],cv2.IMREAD_UNCHANGED)
        # print('1',depth.max(), depth.min(), depth.mean())
##################################################################
        depths = depth.astype(np.float32)
        # print('2',depths.max(), depths.min(), depths.mean())
##################################################################
        image = cv2.undistort(image, K_l, d_l)
        image = torch.as_tensor(image).permute(2, 0, 1)#(3,384,512)change the order of three data.
        # if 309 < t < 320:
        #     image = image * 0
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        depths = depths/5000.0
        valid_mask = depths > 0.1
        depths = np.where(valid_mask, depths, torch.tensor(100))
        valid_mask = depths < 4.0
        depths = np.where(valid_mask, depths, torch.tensor(100))
        depths = 1.0/depths
        # print('3',depths.max(), depths.min(), depths.mean())
        mean = depths.mean()
        depths = depths/ mean
        ####################################
        # nonzero_disps = depths[depths != 0.001]  # 获取非零元素
        # min = np.min(nonzero_disps)  # 获取非零元素的最小值
        # max = depths.max()
        # valid_mask = (depths != 0.001)
        # depths = np.where(valid_mask, depths, min)
        # depths = (depths - min) / (max - min)
        ####################################
        depth = torch.as_tensor(depths)
###############
        # stdv1 = t % 10  # uniform
        # if stdv1 == 0:
        #     image = image * stdv1
        # else:
        #     image = image / stdv1
        # x,y = gps_coords[t]
        yield t, image[None],depth, intrinsics

def image_streams(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([0.262383, -0.953104, -0.005358, 0.002628, 1.163314])
    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        #image = exposure.adjust_gamma(image,4)
        #image = cv2.undistort(image, K_l, d_l)
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((480 * 640) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((480 * 640) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)
        #if t%3 == 0:
            #image = image/2
        #elif t%2 == 0:
            #image = image/3
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--depthdir", type=str, help="path to depth directory")
    parser.add_argument("--gt", type=str, help="path to depth directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1200)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")#2.4
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None
    print(args.imagedir)
    tstamps = []
    for (t, image, depth,intrinsics) in tqdm(image_stream(args.imagedir,args.depthdir, args.calib, args.stride)):
        if t < args.t0:
            continue
        #print(depth)
        if not args.disable_vis:
            show_image(image[0])
        # image = image*0
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        #需要对depth进行处理。
        # print(x,y)
        droid.track(t, image, depth, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.imagedir,args.depthdir, args.calib, args.stride))



    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation
    images_list,depth_list, timestamps = load_tum_data(args.imagedir,'associate.txt')
    # images_list = images_list[::2]
    # depth_list = depth_list[::2]
    # timestamps = timestamps[::2]
    # image_path = os.path.join(args.datapath, 'rgb')
    # images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::2]
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps))
    file_interface.write_tum_trajectory_file(args.imagedir+'.txt', traj_est)
    gt_file = os.path.join(args.imagedir, 'groundtruth.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    #y = traj_ref.reshape(1, -1)
    #np.savetxt("fr1_room.txt",y)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj',
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    #common.plot_result(result, traj_ref,result.trajectories['traj'],traj_ref_full=traj_ref_full)

    print(result)
