
import numbers
import numpy as np
import math as m


class exponential:

	def __init__(self, prefactor = 1, timescale = 1):
		self.prefactor = prefactor
		self.timescale = timescale

	def __call__(self, x):
		return self._prefactor * m.exp(-x / self._timescale)

	@property
	def prefactor(self):
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
		return self._timescale

	@timescale.setter
	def timescale(self, value):
		if isinstance(value, numbers.Number):
			self._timescale = float(value)
		else:
			raise TypeError("Timescale must be a real number. Got: %s" % (
				type(value)))


def cov(data):
	keys = list(data.keys())
	keys = list(filter(lambda x: not x.endswith("_err"), keys))
	# print(keys)
	means = [np.mean(data[_]) for _ in keys]
	cov_ = len(keys) * [None]
	for i in range(len(cov_)): cov_[i] = len(keys) * [0.]
	for i in range(len(cov_)):
		for j in range(len(cov_)):
			key_i = keys[i]
			key_j = keys[j]
			for k in range(len(data[key_i])):
				cov_[i][j] += (data[key_i][k] - means[i]) * (
					data[key_j][k] - means[j])
			cov_[i][j] /= (len(data[key_i]) - 1)
	return cov_

