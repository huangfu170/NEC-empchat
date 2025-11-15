#!/bin/bash
#SBATCH -p batch
#SBATCH -J 测试
#SBATCH -N 1
#SBATCH --gres=gpu:TeslaV100S-PCIE-32GB:1
#SBATCH -o %j.out
#SBATCH -e %j.err

source activate py37

python train_disc_bart.py 