#!/bin/sh


# Activate Anaconda work environment for OpenDrift
source /jet/home/${USER}/.bashrc
source activate torch 

python --version

cd /ocean/projects/cis220078p/vjain1/IFRNet
python generate_flow.py