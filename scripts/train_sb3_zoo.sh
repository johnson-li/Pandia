#!/bin/bash

cd ~/Workspace/rl-baselines3-zoo
PYTHONPATH=~/Workspace/Pandia python train.py --algo sac --env WebRTCEnv-v0 -tb tensorboard --save-freq 1000 --eval-freq 3000 --vec-env subproc --n-eval-envs 1

cd -
