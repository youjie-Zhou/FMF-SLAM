import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import yaml
import argparse
import torch.nn.functional as F
from droid import Droid
import os
import cv2
import numpy as np
def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)
def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[320.0, 320.0, 320.0, 240.0], stereo=False):
    """ image generator """

    # read all png images in folder
    ht0, wd0 = [480, 640]
    images_left = sorted(glob.glob(os.path.join(datapath, 'image_left/*.png')))
    depth_left = sorted(glob.glob(os.path.join(datapath, 'depth_left/*.npy')))
    fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
    data = []
    for t in range(len(images_left)):
        images = cv2.imread(images_left[t])#, (image_size[1], image_size[0])
        # if stereo:
        # depths = [ cv2.resize(cv2.imread(depth_left[t]), (image_size[1], image_size[0])) ]
        depths = np.load(depth_left[t])
        valid_masks = (depths >0.4)
        depths = np.where(valid_masks, depths, 0.4)
####################################################
        # print(images.shape)
        h0, w0, _ = images.shape
        # image = cv2.undistort(image, K_l, d_l)
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        images = cv2.resize(images, (w1, h1))
        # image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        images = torch.as_tensor(images).permute(2, 0, 1)

        depths = torch.as_tensor(depths)
        depths = F.interpolate(depths[None, None], (h1, w1)).squeeze()
        # depth = depth[:h1 - h1 % 8, :w1 - w1 % 8]

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)
####################################################
        depths = 1.0/depths
        depths = depths.numpy()
        images = images.numpy()
        means = np.mean(depths[depths>0.05])
        depths = depths/means
        depths = depths.astype(np.float32)
        images = images.astype(np.float32)
        # np.load
        # print(images.type(),depths.type())
        depths = torch.from_numpy(depths)
        images = torch.from_numpy(images)
        # print('1',images.shape,depths.shape)
        # images = images.permute(2,0,1)
        images = images.unsqueeze(0)
        # print('1',images.shape,depths.shape)
        # depths = depths
        # print('2',images.shape,depths.shape)
        # depths = torch.from_numpy(np.stack(depths, 0)).permute(0, 3, 1, 2)
        # images = images
        # intrinsics = .8 * torch.as_tensor(intrinsics_vec)

        data.append((t, images,depths, intrinsics))

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="datasets/TartanAir")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--id", type=int, default=-1)

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    from data_readers.tartan import test_split
    from evaluation.tartanair_evaluator import TartanAirEvaluator

    if not os.path.isdir("figures"):
        os.mkdir("figures")

    if args.id >= 0:
        test_split = [ test_split[args.id] ]
    # print(args.datapath)
    ate_list = []
    for scene in test_split:
        print("Performing evaluation on {}".format(args.datapath))
        torch.cuda.empty_cache()
        droid = Droid(args)

        scenedir = os.path.join(args.datapath, scene)
        
        for (tstamp, image,depths, intrinsics) in tqdm(image_stream(args.datapath)):
            # image = image * 0
            if not args.disable_vis:
                show_image(image[0])

            droid.track(tstamp, image,depths, intrinsics=intrinsics)

        # fill in non-keyframe poses + global BA
        traj_est = droid.terminate(image_stream(args.datapath))

        ### do evaluation ###

        import evo
        from evo.core.trajectory import PoseTrajectory3D
        from evo.tools import file_interface
        from evo.core import sync
        import evo.main_ape as main_ape
        from evo.core.metrics import PoseRelation

        image_path = os.path.join(args.datapath, 'image_left')
        images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::1]
        tstamps = [float(x.split('/')[-1][:-9]) for x in images_list]#-4

        traj_est = PoseTrajectory3D(
            positions_xyz=traj_est[:,:3],
            orientations_quat_wxyz=traj_est[:,3:],
            timestamps=np.array(tstamps))
        file_interface.write_tum_trajectory_file(args.datapath+'Ours-SLAM.txt', traj_est)
        
        #gt_file = os.path.join(args.imagedir, 'groundtruth.txt')
        poses = np.loadtxt(os.path.join(args.datapath, 'pose_left.txt'), delimiter=' ')#args.imagedir
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
        poses[:,:3] /= 5
        traj_ref = PoseTrajectory3D(
            positions_xyz=poses[:,:3],
            orientations_quat_wxyz=poses[:,3:],
            timestamps=np.array(tstamps))
        #traj_ref = file_interface.read_tum_trajectory_files(gt_file,tstamps)
        #y = traj_ref.reshape(1, -1)
        #np.savetxt("fr1_room.txt",y)

        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
        result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
        #common.plot_result(result, traj_ref,result.trajectories['traj'],traj_ref_full=traj_ref_full)

        print(result)
        file_interface.write_tum_trajectory_file(args.datapath+'groundtruth.txt', traj_ref)
        file_interface.write_tum_trajectory_file(args.imagedir+'groundtruth.txt', traj_ref)
