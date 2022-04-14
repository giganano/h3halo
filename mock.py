r"""
ARGV
----
1) The name of the output file containing the mock sample
"""

from src.utils import piecewise_linear
import numpy as np
import math as m
import vice
from vice.yields.presets import JW20
import sys

# fiducial: 5% precision in O and Fe abundances, 10% in age
FEH_ERR = 0.005
OFE_ERR = 0.005
LOGAGE_ERR = 0.01
NSTARS = 1000
DURATION = 5
H3_UNIVERSE_AGE = 14

# def sfh(t):
# 	return m.exp(-t / 2)
def ifr(t):
	return m.exp(-t / 2)

with vice.singlezone(name = "mock", verbose = True) as sz:
	sz.elements = ["fe", "o"]
	sz.nthreads = 2
	sz.func = ifr
	sz.mode = "ifr"
	# sz.tau_star = 10
	sz.tau_star = piecewise_linear(2)
	sz.tau_star.norm = 50
	sz.tau_star.deltas[0] = 2.5
	sz.tau_star.deltas[1] = 1
	sz.tau_star.slopes[0] = 0
	sz.tau_star.slopes[1] = -48
	sz.tau_star.slopes[2] = 0
	# sz.eta = 25
	sz.eta = 10
	sz.Mg0 = 0
	sz.dt = DURATION / 1000
	sz.run(np.linspace(0, DURATION, 1001), overwrite = True)

with vice.output("mock") as out:
	np.random.seed(0)
	totsfr = sum(out.history["sfr"])
	sfrfrac = [_ / totsfr for _ in out.history["sfr"]]
	indeces = np.random.choice(list(range(len(sfrfrac))), p = sfrfrac,
		size = NSTARS)
	with open(sys.argv[1], 'w') as data:
		# data.write("# [Fe/H]\t[Fe/H]_err\t[O/Fe]\t[O/Fe]_err\n")
		data.write("# [Fe/H]\t[Fe/H]_err\t[O/Fe]\t[O/Fe]_err\t")
		data.write("Log(age/Gyr)\tLog(age/Gyr)_err\n")
		for i in range(len(indeces)):
			feh = out.history["[fe/h]"][indeces[i]]
			ofe = out.history["[o/fe]"][indeces[i]]
			logage = m.log10(out.history["lookback"][indeces[i]] +
				H3_UNIVERSE_AGE - DURATION)
			# logage = m.log10(out.history["lookback"][indeces[i]])
			# age = out.history["lookback"][indeces[i]]
			feh += np.random.normal(scale = FEH_ERR)
			ofe += np.random.normal(scale = OFE_ERR)
			logage += np.random.normal(scale = LOGAGE_ERR)
			# age += np.random.normal(scale = AGE_ERR)
			data.write("%.5e\t%.5e\t" % (feh, FEH_ERR))
			data.write("%.5e\t%.5e\t" % (ofe, OFE_ERR))
			# data.write("%.5e\t%.5e\n" % (age, AGE_ERR))
			if i < 150:
				data.write("%.5e\t%.5e\t" % (logage, LOGAGE_ERR))
				data.write("\n")
			else:
				data.write("nan\tnan\n")
		data.close()
	# out.show("[O/Fe]-[Fe/H]")

