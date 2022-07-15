
from emcee import EnsembleSampler
from scipy.stats import multivariate_normal
import numbers
import numpy as np
import math as m


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


def invcov(errors):
	r"""
	Compute the inverse covariance matrices given the errors on measurements
	in some sample.

	Parameters
	----------
	errors : 2-D ``array-like``
		An mxn list containing the n errors on m data points. If a quantity is
		missing for a given data point (in practice, e.g., ages only available
		for a portion of the sample), then simply swap in a nan for that
		quantity and this routine will invert the covariance matrix without
		this quantity and swap in a row and column of NaN values, ignoring it
		in the likelihood calculation accordingly.
	"""
	nancheck = [m.isnan(_) for _ in errors]
	if any(nancheck):
		smallerrors = list(filter(lambda _: not m.isnan(_), errors))
		smallcov = np.diag([_**2 for _ in smallerrors])
		smallinvcov = np.linalg.inv(smallcov)
		invcov = np.zeros((len(errors), len(errors)))
		nrow = 0
		for i in range(len(invcov)):
			ncol = 0
			if nancheck[i]:
				for j in range(len(invcov[i])): invcov[i][j] = float("nan")
			else:
				for j in range(len(invcov[i])):
					if nancheck[j]:
						invcov[i][j] = float("nan")
					else:
						invcov[i][j] = smallinvcov[nrow][ncol]
						ncol += 1
				nrow += 1
		return invcov
	else:
		cov = np.diag([_**2 for _ in errors])
		return np.linalg.inv(cov)


def chisquared(sample, errors, model, weights):
	norm = sum(weights)
	weights = [_ / norm for _ in weights]
	chisq = 0
	for i in range(len(sample)):
		if any([m.isnan(_) for _ in errors[i]]):
			indeces = []
			for j in range(len(errors[i])):
				if not m.isnan(errors[i][j]): indeces.append(j)
		else:
			indeces = list(range(len(errors[i])))
		sample_ = np.array([sample[i][_] for _ in indeces])
		errors_ = np.array([errors[i][_] for _ in indeces])
		mvn = multivariate_normal(mean = sample_,
			cov = np.diag(errors_**2))
		pdf = len(model) * [0.]
		for j in range(len(pdf)):
			model_ = [model[j][_] for _ in indeces]
			pdf[j] = weights[j] * mvn.pdf(model_)
		idx_match = pdf.index(max(pdf))
		predicted = np.array([model[idx_match][_] for _ in indeces])
		for j in range(len(indeces)):
			chisq += (predicted[j] - sample_[j])**2 / errors_[j]**2
	return chisq


class sinusoid:

	def __init__(self, mean = 0, amplitude = 1, period = 1, shift = 0):
		self.mean = mean
		self.amplitude = amplitude
		self.period = period
		self.shift = shift

	def __call__(self, x):
		return self._mean + self._amplitude * m.sin(
			2 * m.pi * (x - self._shift) / self._period)

	@property
	def mean(self):
		r"""
		Type : ``float``

		Default : 0

		The mean value of the sinusoid in arbitrary units.
		"""
		return self._mean

	@mean.setter
	def mean(self, value):
		if isinstance(value, numbers.Number):
			self._mean = float(value)
		else:
			raise TypeError(
				"Sinusoid mean value must be a real number. Got: %s" % (
					type(value)))

	@property
	def amplitude(self):
		r"""
		Type : ``float``

		Default : 1

		The prefactor of the sinusoid in arbitrary units.
		"""
		return self._amplitude

	@amplitude.setter
	def amplitude(self, value):
		if isinstance(value, numbers.Number):
			self._amplitude = float(value)
		else:
			raise TypeError(
				"Sinusoid amplitude must be a real number. Got: %s" % (
					type(value)))

	@property
	def period(self):
		r"""
		Type : ``float``

		Default : 1

		The period of the sinusoid in arbitrary units.
		"""
		return self._period

	@period.setter
	def period(self, value):
		if isinstance(value, numbers.Number):
			self._period = float(value)
		else:
			raise TypeError(
				"Sinusoid period must be a real number. Got: %s" % (
					type(value)))

	@property
	def shift(self):
		r"""
		Type : ``float``

		Default : 0

		Horizontal shift of the sinusoid in the x-direction in arbitrary
		units.
		"""
		return self._shift

	@shift.setter
	def shift(self, value):
		if isinstance(value, numbers.Number):
			self._shift = float(value)
		else:
			raise TypeError(
				"Sinusoid shift must be a real number. Got: %s" % (
					type(value)))


class gaussian:

	def __init__(self, mean = 0, amplitude = 1, width = 1):
		self.mean = mean
		self.amplitude = amplitude
		self.width = width

	def __call__(self, x):
		return self._amplitude * m.exp(-(x - self._mean)**2 / (
			2 * self._width**2))

	@property
	def mean(self):
		r"""
		Type : ``float``

		Default : 0

		The x-coordinate of the peak of the gaussian in arbitrary units.
		"""
		return self._mean

	@mean.setter
	def mean(self, value):
		if isinstance(value, numbers.Number):
			self._mean = float(value)
		else:
			raise TypeError(
				"Mean of gaussian must be a real number. Got: %s" % (
					type(value)))

	@property
	def amplitude(self):
		r"""
		Type : ``float``

		Default : 1

		The y-coordiate of the gaussian at its peak in arbitrary units.
		"""
		return self._amplitude

	@amplitude.setter
	def amplitude(self, value):
		if isinstance(value, numbers.Number):
			self._amplitude = float(value)
		else:
			raise TypeError(
				"Amplitude of gaussian must be a real number. Got: %s" % (
					type(value)))

	@property
	def width(self):
		r"""
		Type : ``float``

		Default : 1

		The standard deviation of the gaussian in arbitrary units.
		"""
		return self._width

	@width.setter
	def width(self, value):
		if isinstance(value, numbers.Number):
			self._width = float(value)
		else:
			raise TypeError("""\
"Standard deviation of gaussian must be a real number. Got: %s""" % (
				type(value)))


class double_gaussian:

	def __init__(self, mean1 = 0, mean2 = 0, amplitude1 = 1, amplitude2 = 1,
		width1 = 1, width2 = 1):
		self.first = gaussian(mean = mean1, amplitude = amplitude1,
			width = width1)
		self.second = gaussian(mean = mean2, amplitude = amplitude2,
			width = width2)

	def __call__(self, x):
		return self.first.__call__(x) + self.second.__call__(x)


class piecewise_linear:

	def __init__(self, n_knots, norm = 0):
		assert isinstance(n_knots, int)
		assert n_knots > 0
		self.norm = norm
		self._deltas = n_knots * [0.]
		self._slopes = (n_knots + 1) * [0.]

	def __call__(self, x):
		breaks = [0]
		for i in range(len(self._deltas)): breaks.append(
			self._deltas[i] + breaks[-1])
		y = self._norm
		for i in range(self.n_knots):
			if breaks[i] <= x <= breaks[i + 1]:
				y += self._slopes[i] * (x - breaks[i])
				break
			else:
				y += self._slopes[i] * (breaks[i + 1] - breaks[i])
		if x > breaks[-1]: y += self._slopes[-1] * (x - breaks[-1])
		return y

	@property
	def n_knots(self):
		return len(self._deltas)

	@property
	def norm(self):
		return self._norm

	@norm.setter
	def norm(self, value):
		if isinstance(value, numbers.Number):
			self._norm = float(value)
		else:
			raise TypeError("Norm must be a real number. Got: %s" % (
				type(value)))

	@property
	def deltas(self):
		return self._deltas

	@property
	def slopes(self):
		return self._slopes


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


class linear_exponential(exponential):

	def __call__(self, x):
		return x * super().__call__(x)

