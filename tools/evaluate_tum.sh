#!/bin/bash


TUM_PATH=datasets/TUM-RGBD/$seq

evalset=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_tum.py --datapath=$TUM_PATH/$seq --weights=droid.pth --disable_vis $@
done

ETH_PATH=/home/zhou/file8T/qita


    office/hardother/P003
    office/hardother/P004
    office/hardother/P005
    office/hardother/P006
    office/hardother/P007
    # abandonedfactory/other/P006
    # abandonedfactory/other/P007
    # abandonedfactory/other/P008
    # abandonedfactory/other/P009
    # abandonedfactory/other/P010
    # abandonedfactory/other/P011
    # japanesealley/easyother/P005
    # japanesealley/easyother/P004
    # japanesealley/easyother/P007
    # japanesealley/hardother/P003
    # japanesealley/hardother/P004
    # japanesealley/hardother/P005

    #python evaluation_scripts/validate_tartanair.py --datapath=$ETH_PATH/$seq/ --weights=checkpoints/fuseCA+SA+dill/bla_200000.pth --disable_vis $@ # 