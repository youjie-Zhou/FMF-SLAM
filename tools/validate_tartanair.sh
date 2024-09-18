#!/bin/bash


# ETH_PATH=/home/zhou/file8T/qita/abandonedfactory/Hard/depth_left/abandonedfactory/abandonedfactory/Hard
ETH_PATH=/home/zhou/file8T/qita
# ETH_PATH=/home/zhou/file8T/icl
# all "non-dark" training scenes
evalset=(

    # abandonedfactory/other/P006
    # abandonedfactory/other/P007
    # abandonedfactory/other/P008
    # abandonedfactory/other/P009
    # abandonedfactory/other/P010
    abandonedfactory/other/P011

    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P001
    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P004
    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P006
    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P008
    # # office/hardother/P003
    # # office/hardother/P004
    # office/hardother/P005
    # office/hardother/P006
    # office/hardother/P007
    # abandonedfactory_night/easyother/P006
    # abandonedfactory_night/easyother/P007
    # abandonedfactory_night/easyother/P008
    # abandonedfactory_night/easyother/P009
    # abandonedfactory_night/easyother/P010
    # abandonedfactory_night/easyother/P011


    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P001
    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P002
    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P003
    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P004
    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P005
    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P006
    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P007
    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P008
    # endofworld/Easy/depth_left/endofworld/endofworld/Easy/P009

    # abandonedfactory_night/hardother/P008
    # abandonedfactory_night/hardother/P009
    # abandonedfactory_night/hardother/P010
    # # abandonedfactory_night/hardother/P011
    # # abandonedfactory_night/hardother/P012
    # abandonedfactory_night/hardother/P013
    # abandonedfactory_night/hardother/P014

    # japanesealley/easyother/P004
    # japanesealley/easyother/P005
    # japanesealley/easyother/P007
    # japanesealley/hardother/P003
    # japanesealley/hardother/P004
    # japanesealley/hardother/P005

    # oldtown/Easy/depth_left/oldtown/oldtown/Easy/P000


)

for seq in ${evalset[@]}; do
    python evaluation_scripts/validate_tartanair.py --datapath=$ETH_PATH/$seq --weights=checkpoints/fuseCA+SA+dill/bla_200000.pth --disable_vis $@ #--disable_vis 
done



