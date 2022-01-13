#!/bin/bash

#Account and Email Information
#SBATCH -A afarizhandi
#SBATCH --mail-type=end
#SBATCH --mail-user=amirabbaskazemza@boisestate.edu

#SBATCH -J B7-cl2          # job name
#SBATCH -o outputs/results.o%j # output and error file name (%j expands to jobID)
#SBATCH -e outputs/errors.e%j
#SBATCH -p gpu            # queue (partition) -- defq, ipowerq, eduq, gpuq.
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -t 30-00:00:00      # run time (d-hh:mm:ss)
ulimit -v unlimited
ulimit -s unlimited
ulimit -u 1000

module load cuda10.0/toolkit/10.0.130 # loading cuda libraries/drivers 
module load python37          # loading python environment

python3 test_layer_Extract_feature_EfficientNetB7_clean.py
