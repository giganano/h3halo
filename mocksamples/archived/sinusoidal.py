
import sys
sys.path.append("..")
from src.utils import sinusoid
import numpy as np
import math as m
import vice
from vice.yields.presets import JW20
import sys

FEH_ERR = 0.05
OFE_ERR = 0.05
LOGAGE_ERR = 0.1
NSTARS = 500
N_AGES = 150
DURATION = 10
H3_UNIVERSE_AGE = 14

with vice.singlezone(name = "sinusoid", verbose = True) as sz:
	sz.elements = ["fe", "o"]
	sz.nthreads = 2
	sz.func = sinusoid(mean = 1, amplitude = 0.4, period = 2, shift = 0.5)
	sz.mode = "ifr"
	sz.tau_star = 10
	sz.eta = 15
	sz.Mg0 = 0
	sz.dt = DURATION / 1000
	sz.run(np.linspace(0, DURATION, 1001), overwrite = True)

with vice.output("sinusoid") as out:
	np.random.seed(0)
	totsfr = sum(out.history["sfr"])
	sfrfrac = [_ / totsfr for _ in out.history["sfr"]]
	indeces = np.random.choice(list(range(len(sfrfrac))), p = sfrfrac,
		size = NSTARS)
	with open(sys.argv[1], 'w') as data:
		data.write("# [Fe/H]\t[Fe/H]_err\t")
		data.write("[O/Fe]\t[O/Fe]_err\t")
		data.write("Log(age/Gyr)\tLog(age/Gyr)_err\n")
		for i in range(len(indeces)):
			feh = out.history["[fe/h]"][indeces[i]]
			ofe = out.history["[o/fe]"][indeces[i]]
			logage = m.log10(out.history["lookback"][indeces[i]] +
				H3_UNIVERSE_AGE - DURATION)
			feh += np.random.normal(scale = FEH_ERR)
			ofe += np.random.normal(scale = OFE_ERR)
			logage += np.random.normal(scale = LOGAGE_ERR)

			data.write("%.5e\t%.5e\t" % (feh, FEH_ERR))
			data.write("%.5e\t%.5e\t" % (ofe, OFE_ERR))
			if i < N_AGES:
				data.write("%.5e\t%.5e\n" % (logage, LOGAGE_ERR))
			else:
				data.write("nan\tnan\n")
		data.close()

