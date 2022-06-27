#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --ntasks=40
#SBATCH --exclusive
#SBATCH --job-name=gsefit
#SBATCH --account=PCON0003

# start and end the program by echoing the date and time
date
python gsefit.py
date