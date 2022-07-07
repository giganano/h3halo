# cython: language_level = 3, boundscheck = False

import numpy as np
from libc.stdlib cimport malloc, free
cimport _fit_driver

cdef class fit_driver:

	def __cinit__(self, sample, invcovs):
		self._fd = _fit_driver.fit_driver_initialize()

	def __init__(self, sample, invcovs):
		self.sample = sample
		self.invcovs = invcovs

	def __call__(self):
		return _fit_driver.loglikelihood(self._fd[0])

	def __dealloc__(self):
		_fit_driver.fit_driver_free(self._fd)

	@property
	def sample(self):
		_sample = self._fd[0].n_sample * [None]
		for i in range(self._fd[0].n_sample):
			_sample[i] = self._fd[0].n_quantities * [0.]
			for j in range(self._fd[0].n_quantities):
				_sample[i][j] = self._fd[0].sample[i][j]
		return _sample

	@sample.setter
	def sample(self, value):
		if self._fd[0].sample is not NULL: free(self._fd[0].sample)
		self._fd[0].n_sample = <unsigned long> len(value)
		self._fd[0].n_quantities = <unsigned long> len(value[0])
		self._fd[0].sample = <double **> malloc (self._fd[0].n_sample *
			sizeof(double *))
		for i in range(self._fd[0].n_sample):
			self._fd[0].sample[i] = <double *> malloc (
				self._fd[0].n_quantities * sizeof(double))
			for j in range(self._fd[0].n_quantities):
				self._fd[0].sample[i][j] = value[i][j]

	@property
	def invcovs(self):
		_invcov = self._fd[0].n_sample * [None]
		for i in range(self._fd[0].n_sample):
			_invcov[i] = self._fd[0].n_quantities * [None]
			for j in range(self._fd[0].n_quantities):
				_invcov[i][j] = self._fd[0].n_quantities * [0.]
				for k in range(self._fd[0].n_quantities):
					_invcov[i][j][k] = self._fd[0].invcov[i][j][k]
		return _invcov

	@invcovs.setter
	def invcovs(self, value):
		if self._fd[0].invcov is not NULL: free(self._fd[0].invcov)
		self._fd[0].invcov = <double ***> malloc (self._fd[0].n_sample *
			sizeof(double **))
		for i in range(self._fd[0].n_sample):
			self._fd[0].invcov[i] = <double **> malloc (
				self._fd[0].n_quantities * sizeof(double *))
			for j in range(self._fd[0].n_quantities):
				self._fd[0].invcov[i][j] = <double *> malloc (
					self._fd[0].n_quantities * sizeof(double))
				for k in range(self._fd[0].n_quantities):
					self._fd[0].invcov[i][j][k] = value[i][j][k]

	@property
	def model(self):
		_model = self._fd[0].n_model * [None]
		for i in range(self._fd[0].n_model):
			_model[i] = self._fd[0].n_quantities * [0.]
			for j in range(self._fd[0].n_quantities):
				_model[i][j] = self._fd[0].model[i][j]
		return _model

	@model.setter
	def model(self, value):
		if self._fd[0].model is not NULL: free(self._fd[0].model)
		self._fd[0].n_model = <unsigned long> len(value)
		self._fd[0].model = <double **> malloc (self._fd[0].n_model *
			sizeof(double *))
		for i in range(self._fd[0].n_model):
			self._fd[0].model[i] = <double *> malloc (
				self._fd[0].n_quantities * sizeof(double))
			for j in range(self._fd[0].n_quantities):
				self._fd[0].model[i][j] = value[i][j]

	@property
	def weights(self):
		return [self._fd[0].weights[_] for _ in range(self._fd[0].n_model)]

	@weights.setter
	def weights(self, value):
		if self._fd[0].weights is not NULL: free(self._fd[0].weights)
		self._fd[0].weights = <double *> malloc (self._fd[0].n_model *
			sizeof(double))
		norm = sum(value)
		for i in range(self._fd[0].n_model): 
			if value[i] >= 0:
				self._fd[0].weights[i] = value[i] / norm
			else:
				raise ValueError("Negative weight at index %d." % (i))

