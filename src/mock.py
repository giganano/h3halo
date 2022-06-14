
import numpy as np
import math as m
import random
import vice
# from vice.yields.presets import JW20
import sys


def create_mock(sz, outfile, n_stars = 500, n_ages = 100, feh_err = 0.05,
	ofe_err = 0.05, logage_err = 0.1, duration = 10, cosmological_age = 13.2,
	seed = 0):

	r"""
	Generate a mock sample from a onezone model.

	Parameters
	----------
	sz : vice.singlezone
		The singlezone object in VICE to run and produce the mock sample
		output from.
	outfile : str
		The name of the output file to store the mock data in.
	n_stars : int [default : 500]
		The number of stellar populations to draw from the one-zone model as
		mock data.
	n_ages : int [default : 100]
		The number of ages to attach to the stellar population.
	feh_err : float [default : 0.05]
		The artificial error on the [Fe/H] measurement.
	ofe_err : float [default : 0.05]
		The artificial error on the [O/Fe] measurement.
	logage_err : float [default : 0.1]
		The artificial error on log(age/Gyr)
	duration : float [default : 10]
		The timescale over which to integrate the one-zone model.
	cosmological_age : float [default : 13.2]
		Lookback time to the onset of star formation in the model in Gyr.
	seed : int [default : 0]
		Pseudo-random number generator seed to pass to numpy.random.seed.
	"""

	assert isinstance(sz, vice.singlezone)
	assert n_stars >= n_ages >= 0
	assert 0 < duration <= cosmological_age
	assert feh_err > 0
	assert ofe_err > 0
	assert logage_err > 0

	n_outputs = int(duration / sz.dt) + 1
	with sz.run(np.linspace(0, duration, n_outputs), overwrite = True,
		capture = True) as out:
		random.seed(seed)
		totsfr = sum(out.history["sfr"])
		sfrfrac = [_ / totsfr for _ in out.history["sfr"]]
		indeces = np.random.choice(list(range(len(sfrfrac))), p = sfrfrac,
			size = n_stars)
		with open(outfile, 'w') as data:
			data.write("# [Fe/H]\t[Fe/H]_err\t[O/Fe]\t[O/Fe]_err\t")
			data.write("Log(age/Gyr)\tLog(age/Gyr)_err\n")
			for i in range(len(indeces)):
				feh = out.history["[fe/h]"][indeces[i]]
				ofe = out.history["[o/fe]"][indeces[i]]
				logage = m.log10(out.history["lookback"][indeces[i]] +
					cosmological_age - duration)
				feh += np.random.normal(scale = feh_err)
				ofe += np.random.normal(scale = ofe_err)
				logage += np.random.normal(scale = logage_err)

				data.write("%.5e\t%.5e\t" % (feh, feh_err))
				data.write("%.5e\t%.5e\t" % (ofe, ofe_err))
				if i < n_ages:
					data.write("%.5e\t%.5e\n" % (logage, logage_err))
				else:
					data.write("nan\tnan\n")
			data.close()


