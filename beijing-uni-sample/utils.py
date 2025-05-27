
from emcee import EnsembleSampler
from scipy.stats import multivariate_normal
import numpy as np
import math as m
import trackstar
import numbers
import vice
import os


def get_sample():
	raw = np.genfromtxt("%s/gse-with-seismo.csv" % (
		os.path.dirname(os.path.abspath(__file__))), delimiter = ',')
	with open("gse-with-seismo.csv", 'r') as f:
		header = f.readline()[1:-1] # chop off leading '#' and trailing '\n'
		header = header.split(',')
	sample = {}
	for i in range(len(header)):
		sample[header[i]] = raw[:, i + 1]
	sample = vice.dataframe(sample)
	return sample


def savechain(sampler, filename):
	r"""
	Save the MCMC chain to an output file.

	Parameters
	----------
	sampler : ``emcee.EnsembleSampler``
		The sampler object from ``emcee`` containing the Markov Chain.
	filename : ``str``
		The name of the file to save the Markov Chain to.
	"""
	assert isinstance(sampler, EnsembleSampler)
	samples = sampler.get_chain()
	logprob = sampler.get_log_prob()
	samples = np.concatenate(tuple([samples[i] for i in range(len(samples))]))
	logprob = np.concatenate(tuple([logprob[i] for i in range(len(logprob))]))
	logprob = [[logprob[_]] for _ in range(len(logprob))]
	out = np.append(samples, logprob, axis = 1)
	af = sum(sampler.acceptance_fraction) / sampler.nwalkers
	np.savetxt(filename, out, fmt = "%.5e",
		header = "acceptance fraction: %.5e" % (af))


class singlezone_mcmc:

	def __init__(self, data):
		self.sample = trackstar.sample(data)
		self.sz = vice.singlezone()

	# def __call__(self, output):

	# def get_track(self):
	# 	track = {}
	# 	output = vice.output(self.sz.name)
	# 	for elem in output.elements:
	# 		if elem == "fe":
	# 			track["[fe/h]"] = output.history["[fe/h]"]
	# 		else:
	# 			track["[%s/fe]" % (elem)] = output.history["[%s/fe]" % (elem)]
	# 	track["lookback"] = output.history["lookback"]
	# 	return trackstar.track(track)



class exponential:

	r"""
	A simple exponential function in an arbitrary x coordinate.

	Parameters & Attributes
	-----------------------
	prefactor : ``float`` [default: 1]
		A parameter setting the overall normalization of the exponential.
	timescale : ``float`` [default : 1]
		A parameter describing the rate of exponential decay in the arbitrary
		x coordinate.

	Call this object with the arbitrary x-coordinate as the single parameter
	and it will return the value of

	.. math:: f(x) = Ae^{-t/\tau}

	where :math:`A` is the attribute ``prefactor`` and :math:`\tau` is the
	attribute ``timescale``.
	"""

	def __init__(self, prefactor = 1, timescale = 1):
		self.prefactor = prefactor
		self.timescale = timescale

	def __call__(self, x):
		return self._prefactor * m.exp(-x / self._timescale)

	@property
	def prefactor(self):
		r"""
		Type : ``float``

		Default : 1

		A parameter describing the overall normalization of the exponential.
		"""
		return self._prefactor

	@prefactor.setter
	def prefactor(self, value):
		if isinstance(value, numbers.Number):
			self._prefactor = float(value)
		else:
			raise TypeError("Prefactor must be a real number. Got: %s" % (
				type(value)))

	@property
	def timescale(self):
		r"""
		Type : ``float``

		Default : 1

		A parameter describing the e-folding rate of the exponential.
		"""
		return self._timescale

	@timescale.setter
	def timescale(self, value):
		if isinstance(value, numbers.Number):
			self._timescale = float(value)
		else:
			raise TypeError("Timescale must be a real number. Got: %s" % (
				type(value)))

