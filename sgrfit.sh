#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=40
#SBATCH --exclusive
#SBATCH --job-name=sgrfit
#SBATCH --account=PCON0003

date
python sgrfit.py ./data/sgr/sgrchem.dat ./data/sgr/sgrchem_102k4.out sgr
date

