#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=40
#SBATCH --exclusive
#SBATCH --job-name=degeneracy
#SBATCH --account=PCON0003

# start and end the program by echoing the date and time
date

# sample size mock fits
# python mockfit.py ./mocksamples/n2000.dat ./mocksamples/n2000_256k.out n2000_
# python mockfit.py ./mocksamples/n1000.dat ./mocksamples/n1000_256k.out n1000_
python mockfit.py ./mocksamples/fiducial.dat ./mocksamples/fiducial_512k.out fiducial_
# python mockfit.py ./mocksamples/n200.dat ./mocksamples/n200_256k.out n200_
# python mockfit.py ./mocksamples/n100.dat ./mocksamples/n100_256k.out n100_
# python mockfit.py ./mocksamples/n50.dat ./mocksamples/n50_256k.out n50_
# python mockfit.py ./mocksamples/n20.dat ./mocksamples/n20_25k6.out n20_

# Abundance precision mock fits
# python mockfit.py ./mocksamples/ab_err_0p5.dat ./mocksamples/ab_err_0p5_25k6.out ab_err_0p5_
# python mockfit.py ./mocksamples/ab_err_0p2.dat ./mocksamples/ab_err_0p2_25k6.out ab_err_0p2_
# python mockfit.py ./mocksamples/ab_err_0p1.dat ./mocksamples/ab_err_0p1_25k6.out ab_err_0p1_
# python mockfit.py ./mocksamples/ab_err_0p02.dat ./mocksamples/ab_err_0p02_25k6.out ab_err_0p02_
# python mockfit.py ./mocksamples/ab_err_0p01.dat ./mocksamples/ab_err_0p01_25k6.out ab_err_0p01_

# Age precision mock fits
# python mockfit.py ./mocksamples/age_err_1p0.dat ./mocksamples/age_err_1p0_25k6.out age_err_1p0_
# python mockfit.py ./mocksamples/age_err_0p5.dat ./mocksamples/age_err_0p5_25k6.out age_err_0p5_
# python mockfit.py ./mocksamples/age_err_0p2.dat ./mocksamples/age_err_0p2_25k6.out age_err_0p2_
# python mockfit.py ./mocksamples/age_err_0p05.dat ./mocksamples/age_err_0p05_25k6.out age_err_0p05_
# python mockfit.py ./mocksamples/age_err_0p02.dat ./mocksamples/age_err_0p02_25k6.out age_err_0p02_

# Age availability mock fits
# python mockfit.py ./mocksamples/agefrac_0p0.dat ./mocksamples/agefrac_0p0_25k6.out agefrac_0p0_
# python mockfit.py ./mocksamples/agefrac_0p1.dat ./mocksamples/agefrac_0p1_25k6.out agefrac_0p1_
# python mockfit.py ./mocksamples/agefrac_0p3.dat ./mocksamples/agefrac_0p3_25k6.out agefrac_0p3_
# python mockfit.py ./mocksamples/agefrac_0p4.dat ./mocksamples/agefrac_0p4_25k6.out agefrac_0p4_
# python mockfit.py ./mocksamples/agefrac_0p5.dat ./mocksamples/agefrac_0p5_25k6.out agefrac_0p5_
# python mockfit.py ./mocksamples/agefrac_0p6.dat ./mocksamples/agefrac_0p6_25k6.out agefrac_0p6_
# python mockfit.py ./mocksamples/agefrac_0p7.dat ./mocksamples/agefrac_0p7_25k6.out agefrac_0p7_
# python mockfit.py ./mocksamples/agefrac_0p8.dat ./mocksamples/agefrac_0p8_25k6.out agefrac_0p8_
# python mockfit.py ./mocksamples/agefrac_0p9.dat ./mocksamples/agefrac_0p9_25k6.out agefrac_0p9_
# python mockfit.py ./mocksamples/agefrac_1p0.dat ./mocksamples/agefrac_1p0_25k6.out agefrac_1p0_

# example fit with all yields as free parameters
# python mockfit.py ./mocksamples/fiducial.dat ./mocksamples/degeneracy_512k.out degeneracy_

date

