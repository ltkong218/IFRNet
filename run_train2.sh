#!/bin/sh


# Activate Anaconda work environment for OpenDrift
source /jet/home/${USER}/.bashrc
source activate torch 

python --version

cd /ocean/projects/cis220078p/vjain1/IFRNet
export OMP_NUM_THREADS=1
nohup bash watch_cpu.sh & > gpu2.log
python -m torch.distributed.launch --nproc_per_node=8 train_vimeo90k.py --world_size 8 --model_name 'IFRNet' --epochs 300 --batch_size 55  --lr_start 1e-4 --lr_end 1e-5