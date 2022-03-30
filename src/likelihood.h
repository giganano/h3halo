
#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H

typedef struct fit_driver {

	/*
	 * n_model : the number of data vectors in the model
	 * n_sample : the number of data vectors in the sample
	 * n_quantities : the dimensionality of the sample
	 * model : an n_model x n_quantities array containing the model vectors
	 * sample : an n_sample x n_quantities array containing the data
	 * invcov : an n_sample length array containing the n_quantities x
	 * 		n_quantities inverse covariance matrices for each individual
	 * 		point in the sample.
	 * weights : an n_model length array of weights to assign to each model
	 * 		data vector.
	 */

	unsigned long n_model;
	unsigned long n_sample;
	unsigned short n_quantities;
	double **model;
	double **sample;
	double ***invcov;
	double *weights;

} FIT_DRIVER;

extern FIT_DRIVER *fit_driver_initialize(void);
extern void fit_driver_free(FIT_DRIVER *fd);
extern double loglikelihood(FIT_DRIVER fd);

#endif /* LIKELIHOOD_H */

