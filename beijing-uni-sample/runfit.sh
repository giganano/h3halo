#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntaska=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name="gsefit-prelim"
#SBATCH --output="%j.log"

date
python gsefit.py
date
