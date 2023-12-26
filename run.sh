#!/bin/bash
cd ~/DC
for target_v in 0.05
do
for multi in True
do
for _ in {0..9}
do
for seed in 4
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset maze2d-large-v1 \
                                                           --control position \
                                                           --increasing_condition False \
                                                           --target_v $target_v \
                                                           --seed $seed \
                                                           --diffusion_epoch 999999 \
                                                           --multi $multi
done
done
done
done