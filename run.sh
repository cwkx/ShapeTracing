#!/bin/bash
# This line is required to inform the Linux
#command line to parse the script using
#the bash shell

# Instructing SLURM to locate and assign
#X number of nodes with Y number of
#cores in each node.
# X,Y are integers. Refer to table for
#various combinations
#SBATCH -N 1
#SBATCH -c 4

#SBATCH --mem=28G
# Governs the run time limit and
# resource limit for the job. Please pick values
# from the partition and QOS tables below
#for various combinations
#SBATCH --gres=gpu:1
#SBATCH -p "res-gpu-small"
#SBATCH --qos="short"
#SBATCH -t 2-0

# Source the bash profile (required to use the module command)
source /etc/profile
module load cuda/10.1-cudnn7.6

# make sure pipenv is installed for me on client nodes
pip3 install --user pipenv
# Install/check install of packages
export VIS_PORT='9003'
python3 -m pipenv install --skip-lock
python3 -m pipenv run python3 -u -Wignore k_iso_opt.py $@