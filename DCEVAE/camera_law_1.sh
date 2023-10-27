#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=camera_law_1 --gpus-per-node=1 --ntasks-per-node=32
#SBATCH --account=PAS2563

#python3 main.py --dataset law --a_y 1 --a_r 1 --a_d 1 --a_f 0.2 --a_h 0.4 --u_kl 1 --retrain True --n_epochs 2000 --lr 1e-3 --normalize True --run 1
python3 main.py --dataset law --a_y 1 --a_r 1 --a_d 1 --a_f 0.2 --a_h 0.4 --u_kl 1 --retrain True --n_epochs 2000 --lr 1e-3 --use_label True --normalize True --run 1