#!/bin/bash
#SBATCH -p long
#SBATCH -J train_emotion_cls_bs64
#SBATCH -N 1
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH -o %j.out
#SBATCH -e %j.err

source activate py37

python EmotionClassifacation.py
