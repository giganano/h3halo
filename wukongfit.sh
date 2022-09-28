#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH --ntasks=40
#SBATCH --exclusive
#SBATCH --job-name=wukongfit
#SBATCH --account=PAS1140

date
python wukongfit.py
date
