#!/bin/bash


# ETH_PATH=/home/zhou/file8T/qita/abandonedfactory/Hard/depth_left/abandonedfactory/abandonedfactory/Hard
ETH_PATH=/home/zhou/file8T/data

evalset=(
    deer_run
    deer_walk
    deer_MAV_Slow
    deer_MAV_Fast
    diamond_MAV_Fast
    deer_ground_robot
    diamond_walk
    diamond_ground_robot
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/demodepthicl.py  --imagedir=$ETH_PATH/$seq --gt=$ETH_PATH/$seq --weights=checkpoints/fuseCA+SA+dill/bla_200000.pth --disable_vis  $@ #
done



