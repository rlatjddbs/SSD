#!/bin/bash
cd ~/DC

/home/sykim/anaconda3/envs/diffuser10/bin/python eval.py --cid $1 --pid $2 --dataset maze2d-umaze-v1 \
                                                           --control position \
                                                       
