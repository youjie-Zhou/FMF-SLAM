#!/bin/bash


ETH_PATH=/home/zhou/file8T/qita

evalset=(
    japanesealley/easyother/P005
    japanesealley/easyother/P004
    japanesealley/easyother/P007
    japanesealley/hardother/P003
    japanesealley/hardother/P004
    japanesealley/hardother/P005
    # office/hardother/P003
    # office/hardother/P004
    # office/hardother/P005
    # office/hardother/P006
    # office/hardother/P007
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/validate_tartanair.py --datapath=$ETH_PATH/$seq/ --weights=checkpoints/fuseCA+SA+dill/bla_200000.pth --disable_vis $@ 
done

