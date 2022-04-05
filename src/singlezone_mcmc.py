
from multiprocessing import Pool
from emcee import EnsembleSampler
from . import fit_driver
from .utils import invcov
import numpy as np
import math as m
import vice
import sys
import os


class mcmc:

	r"""
	A base class for using MCMC methods to find best-fit parameters to
	one-zone models of chemical evolution using VICE's ``singlezone`` object.

	Creating an instance of this object will store the data and inverse
	covariance matrices as properties. If there are quantities missing for
	some portion of the sample, include the error on the missing quantity
	as a NaN and this object will neglect it in likelihood calculations
	accordingly. The keys of this dictionary should match the keys of the
	``vice.output`` object for automatic likelihood computations.

	To use this object, subclass it and give the ``__call__`` function a
	parameter ``walker`` which will represent the model parameters passed from
	``emcee``. Make the necessary updates to the attribute ``sz`` (the instance
	of ``vice.singlezone`` used to run the integration) and run the model,
	then simply call the base ``__call__`` function with the ``vice.output``
	object produced from the model and it will automatically compute the
	likelihood.
	"""

	def __init__(self, data):
		assert isinstance(data, dict)
		self._quantities = list(data.keys())
		self._quantities = list(filter(lambda _: not _.endswith("_err"),
			self.quantities))
		sample = np.array([data[key] for key in self._quantities]).T
		errors = np.array(
			[data["%s_err" % (key)] for key in self._quantities]).T
		invcovs = len(sample) * [None]
		for i in range(len(sample)): invcovs[i] = invcov(errors[i])
		self._fd = fit_driver(sample, invcovs)
		self._sz = vice.singlezone()

	def __call__(self, output):
		model = []
		for key in self._quantities:
			if key == "lookback":
				model.append([m.log10(_) for _ in output.history[key][:-1]])
			else:
				model.append(output.history[key][1:])
		model = np.array(model).T
		weights = output.history["sfr"][1:]
		norm = sum(weights)
		weights = [_ / norm for _ in weights]

		self._fd.model = model
		self._fd.weights = weights
		return self._fd()

	@property
	def sz(self):
		r"""
		Type : ``vice.singlezone``

		The one-zone model with desired parameterization built into its
		attributes. This class is designed to find best-fit values for those
		parameters given the data set assigned to this class.
		"""
		return self._sz

	@property
	def fd(self):
		r"""
		Type : ``src.fit_driver``

		The ``fit_driver`` object (implemented in src/_fit_driver.pyx) which
		computes :math:`\ln L` for a given set of model predictions.
		"""
		return self._fd

	@property
	def quantities(self):
		r"""
		Type : ``list`` [elements of type ``str``]

		A list containing identifiers of each of the quantities built into the
		data sample.
		"""
		return self._quantities


