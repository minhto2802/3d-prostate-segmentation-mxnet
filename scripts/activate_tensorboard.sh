#!/usr/bin/env bash
cd /media/data1/minh/projects/PROMISE2012_v2/results/ThirdAttempt
conda activate tensorflow36
tensorboard --logdir=. --host=127.0.0.1 --port=6009
