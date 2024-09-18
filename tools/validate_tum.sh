#!/bin/bash


# ETH_PATH=/home/zhou/file8T/qita/abandonedfactory/Hard/depth_left/abandonedfactory/abandonedfactory/Hard
# ETH_PATH=/home/zhou/ssd/dataset/TUM
ETH_PATH=/home/zhou/file8T/biyedataset
evalset=(

    # rgbd_dataset_freiburg1_360
    # rgbd_dataset_freiburg1_desk
    # rgbd_dataset_freiburg1_desk2
    # rgbd_dataset_freiburg1_floor
    # rgbd_dataset_freiburg1_plant
    # rgbd_dataset_freiburg1_room
    # rgbd_dataset_freiburg1_rpy
    # rgbd_dataset_freiburg1_teddy
    # rgbd_dataset_freiburg1_xyz
    # rgbd_dataset_freiburg2_desk
    # rgbd_dataset_freiburg2_pioneer_360
    # rgbd_dataset_freiburg2_pioneer_slam
    # rgbd_dataset_freiburg2_pioneer_slam3 
    # rgbd_dataset_freiburg2_xyz

    # rgbd_dataset_freiburg3_cabinet
    # rgbd_dataset_freiburg3_large_cabinet
    # rgbd_dataset_freiburg3_long_office_household
    # rgbd_dataset_freiburg3_nostructure_notexture_near_withloop
    # rgbd_dataset_freiburg3_nostructure_texture_far
    # rgbd_dataset_freiburg3_nostructure_texture_near_withloop
    # rgbd_dataset_freiburg3_structure_texture_far
    # rgbd_dataset_freiburg3_structure_texture_near
    # 20240329_203637
    # 20240329_215934
    # 20240331_134304
    # 20240331_134413
    # huojia
    # indoor_loop1
    # indoor_loop2
    # indoor_loop3
    # huojia2
    # indoor_loop4
    # huojiadark
    # indoor_loop_dark
    # huojia3
    # indoor_loop5
    # changeexp1
    # changeexp2
    # changeexp3
    # changeexp4
    # out4
    # out3
    # out5
    # changeexp5
    # changeexp6
    # out6dark
    # out7dark
    # indoor-standard2
    # indoor-lightchanging
    # indoor-lightchanging2
    # indoor-dark2
    # indoor-dark3
    # indoor-dark4
    indoor-dark5
    # indoor-dark6
    # indoor-dark7
    # indoor_loop1
    # indoor_loop_dark

)

for seq in ${evalset[@]}; do
    python evaluation_scripts/demodepthtum.py  --imagedir=$ETH_PATH/$seq --gt=$ETH_PATH/$seq --weights=checkpoints/fuseCA+SA+dill/bla_200000.pth --disable_vis $@ #  --disable_vis
done



