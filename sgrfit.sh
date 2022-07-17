#!/bin/bash
#SBATCH --time=7:00:00
#SBATCH --ntasks=40
#SBATCH --exclusive
#SBATCH --job-name=sgrfit
#SBATCH --account=PCON0003

date
python sgrfit.py ./data/sgr/sgrchem.dat ./data/sgr/sgrchem_gaussians_1m024k.out sgr
date

