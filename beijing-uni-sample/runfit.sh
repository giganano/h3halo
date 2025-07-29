#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name="gsefit-prelim"
#SBATCH --output="%j.log"
#SBATCH --time=24:00:00

date
python gsefit-withprior.py
date
