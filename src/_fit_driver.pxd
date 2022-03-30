# cython: language_level = 3, boundscheck = False

cdef extern from "./likelihood.h":
	ctypedef struct FIT_DRIVER:
		unsigned long n_model
		unsigned long n_sample
		unsigned long n_quantities
		double **model
		double **sample
		double ***invcov
		double *weights

	FIT_DRIVER *fit_driver_initialize()
	void fit_driver_free(FIT_DRIVER *fd)
	double loglikelihood(FIT_DRIVER fd)


cdef class fit_driver:
	cdef FIT_DRIVER *_fd


