/*
 * Log likelihood calculations for MCMC routines.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "likelihood.h"

static double **multiply_matrices(double **matrix1, double **matrix2,
	unsigned short l, unsigned short m, unsigned short n);
static double **transpose(double **matrix, unsigned short m, unsigned short n);
static double chi_squared(FIT_DRIVER fd, unsigned long model_idx,
	unsigned long sample_idx);


extern FIT_DRIVER *fit_driver_initialize(void) {

	FIT_DRIVER *fd = (FIT_DRIVER *) malloc (sizeof(FIT_DRIVER));
	fd -> n_model = 0ul;
	fd -> n_sample = 0ul;
	fd -> n_quantities = 0u;
	fd -> model = NULL;
	fd -> sample = NULL;
	fd -> invcov = NULL;
	fd -> weights = NULL;
	return fd;

}


extern void fit_driver_free(FIT_DRIVER *fd) {

	if (fd != NULL) {

		if ((*fd).model != NULL) {
			unsigned long i;
			for (i = 0ul; i < (*fd).n_model; i++) free(fd -> model[i]);
			free(fd -> model);
			fd -> model = NULL;
		} else {}

		if ((*fd).sample != NULL) {
			unsigned long i;
			for (i = 0ul; i < (*fd).n_sample; i++) free(fd -> sample[i]);
			free(fd -> sample);
			fd -> sample = NULL;
		} else {}

		if ((*fd).invcov != NULL) {
			unsigned long i;
			for (i = 0ul; i < (*fd).n_sample; i++) {
				unsigned short j;
				for (j = 0u; j < (*fd).n_quantities; j++) {
					free(fd -> invcov[i][j]);
				}
				free(fd -> invcov[i]);
			}
			free(fd -> invcov);
			fd -> invcov = NULL;
		} else {}

		if ((*fd).weights != NULL) {
			free(fd -> weights);
			fd -> weights = NULL;
		} else {}

		free(fd);
		fd = NULL;

	} else {}

}


extern double loglikelihood(FIT_DRIVER fd) {

	double result = 0;
	unsigned long i, j;
	for (i = 0ul; i < fd.n_sample; i++) {
		double s = 0;
		for (j = 0ul; j < fd.n_model; j++) {
			s += fd.weights[j] * exp(-0.5 * chi_squared(fd, j, i));
		}
		result += log(s);
	}
	return result;

}


static double chi_squared(FIT_DRIVER fd, unsigned long model_idx,
	unsigned long sample_idx) {

	unsigned short i;
	double **delta = (double **) malloc (sizeof(double *));
	delta[0] = (double *) malloc (fd.n_quantities * sizeof(double));
	for (i = 0u; i < fd.n_quantities; i++) {
		delta[0][i] = fd.model[model_idx][i] - fd.sample[sample_idx][i];
	}

	double **delta_t = transpose(delta, 1u, fd.n_quantities);
	double **first_product = multiply_matrices(delta, fd.invcov[sample_idx],
		1u, fd.n_quantities, fd.n_quantities);
	double **second_product = multiply_matrices(first_product, delta_t,
		1u, fd.n_quantities, 1u);
	double chi_squared = second_product[0][0];
	free(delta[0]);
	free(delta);
	for (i = 0u; i < fd.n_quantities; i++) free(delta_t[i]);
	free(delta_t);
	free(first_product[0]);
	free(first_product);
	free(second_product[0]);
	free(second_product);
	return chi_squared;

}


/*
 * Transpose an mxn matrix.
 */
static double **transpose(double **matrix, unsigned short m, unsigned short n) {

	unsigned short i, j;
	double **trans = (double **) malloc (n * sizeof(double *));
	for (i = 0u; i < n; i++) {
		trans[i] = (double *) malloc (m * sizeof(double));
		for (j = 0u; j < m; j++) {
			trans[i][j] = matrix[j][i];
		}
	}

	return trans;

}


/*
 * Multiply two matrices together. matrix1 is an lxm matrix and matrix2 is an
 * mxn matrix. The result is an lxn matrix.
 */
static double **multiply_matrices(double **matrix1, double **matrix2,
	unsigned short l, unsigned short m, unsigned short n) {

	unsigned short i, j, k;
	double **result = (double **) malloc (l * sizeof(double *));
	for (i = 0u; i < l; i++) {
		result[i] = (double *) malloc (n * sizeof(double));
		for (j = 0u; j < n; j++) {
			result[i][j] = 0;
			for (k = 0u; k < m; k++) {
				result[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}

	return result;

}

