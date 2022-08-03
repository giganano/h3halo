#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=40
#SBATCH --exclusive
#SBATCH --job-name=wukongfit
#SBATCH --account=PCON0003

date
python wukongfit.py
date
