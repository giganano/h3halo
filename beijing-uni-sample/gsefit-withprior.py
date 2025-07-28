
from multiprocessing import Pool
from emcee import EnsembleSampler
from utils import singlezone_mcmc, exponential, savechain
import numpy as np
import math as m
import trackstar
import numbers
import vice
import time
import sys
sys.path.append("..")
from gsefit import gsefit as gsefit_base
import os
from utils import get_sample
# import yields

vice.yields.ccsne.settings['mg'] = 1.2e-4

OUTFILE = "./gsefit_test.out"
MODEL_BASENAME = "gsefit"

N_PROC = 4
N_TIMESTEPS = 500
N_WALKERS = 256
N_BURNIN = 1000
N_ITERS = 2000
COSMOLOGICAL_AGE = 13.2
N_DIM = 6


# emcee walker parameters (exponential IFR):
#
# 0. Infall history e-folding timescale
# 1. Mass loading factor
# 2. SFE timescale in Gyr
# 3. Duration of star formation in Gyr
# 4. CCSN yield of Fe
# 5. SN Ia yield of Fe


class gsefit(gsefit_base):

	def __init__(self, data):
		super().__init__(data)
		self.sz.elements = ["fe", "mg"]

	def __call__(self, walker):
		logl = super().__call__(walker)
		if np.isinf(logl) or np.isnan(logl): return -float("inf")
		logl *= self.johnson2023_prior(walker)
		return logl

	@staticmethod
	def johnson2023_prior(walker):
		factor = 1
		factor *= np.exp(-(walker[0] - 1.01)**2 / (2 * 0.13**2))
		factor *= np.exp(-(walker[1] - 8.84)**2 / (2 * 0.86**2))
		factor *= np.exp(-(walker[2] - 16.08)**2 / (2 * 1.29**2))
		factor *= np.exp(-(walker[3] - 5.40)**2 / (2 * 0.31**2))
		factor *= np.exp(-(walker[4] - 7.78e-4)**2 / (2 * 0.37e-4**2))
		factor *= np.exp(-(walker[5] - 1.23e-3)**2 / (2 * 0.11e-3**2))
		return factor


# class gsefit(singlezone_mcmc):

# 	def __init__(self, data):
# 		super().__init__(data)
# 		self.track = None
# 		self.sz.elements = ["fe", "mg"]
# 		self.sz.func = exponential(prefactor = 1000)
# 		self.sz.mode = "ifr"
# 		self.sz.Mg0 = 0


# 	def __call__(self, walker):
# 		if any([_ < 0 for _ in walker]): return -float("inf")
# 		if walker[3] > COSMOLOGICAL_AGE: return -float("inf")
# 		prior = self.johnson2023_prior(walker)
# 		print("walker: [%.2f, %.2f, %.2f, %.2f, %.2e, %.2e]" % (
# 			walker[0], walker[1], walker[2], walker[3], walker[4], walker[5]))
# 		self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())
# 		self.sz.func.timescale = walker[0]
# 		self.sz.eta = walker[1]
# 		self.sz.tau_star = walker[2]
# 		self.sz.dt = walker[3] / N_TIMESTEPS
# 		vice.yields.ccsne.settings['fe'] = walker[4]
# 		vice.yields.sneia.settings['fe'] = walker[5]
# 		output = self.sz.run(
# 			np.linspace(0, walker[3], N_TIMESTEPS + 1),
# 			overwrite = True,
# 			capture = True)
# 		diff = COSMOLOGICAL_AGE - walker[3]
# 		model = {}
# 		for elem in self.sz.elements:
# 			if elem == "fe":
# 				model["[fe/h]"] = output.history["[fe/h]"][:N_TIMESTEPS]
# 			else:
# 				model["[%s/fe]" % (elem)] = output.history["[%s/fe]" % (
# 					elem)][:N_TIMESTEPS]
# 		model["logage"] = [m.log10(
# 			_ + diff) for _ in output.history["lookback"]][:N_TIMESTEPS]
# 		if self.track is not None:
# 			for key in model.keys():
# 				self.track[key] = model[key]
# 		else:
# 			self.track = trackstar.track(model)
# 		result = self.sample.loglikelihood(self.track)
# 		if m.isnan(result): result = -float("inf")
# 		return prior * result

	# @staticmethod
	# def johnson2023_prior(walker):
	# 	factor = 1
	# 	factor *= np.exp(-(walker[0] - 1.01)**2 / (2 * 0.13**2))
	# 	factor *= np.exp(-(walker[1] - 8.84)**2 / (2 * 0.86**2))
	# 	factor *= np.exp(-(walker[2] - 16.08)**2 / (2 * 1.29**2))
	# 	factor *= np.exp(-(walker[3] - 5.40)**2 / (2 * 0.31**2))
	# 	factor *= np.exp(-(walker[4] - 7.78e-4)**2 / (2 * 0.37e-4**2))
	# 	factor *= np.exp(-(walker[5] - 1.23e-3)**2 / (2 * 0.11e-3**2))
	# 	return factor


if __name__ == "__main__":
	sample = get_sample().filter("mgfe_e", ">", 0)
	logl = gsefit({
		"[fe/h]": sample["feh"],
		"[mg/fe]": sample["mgfe"],
		"[fe/h]_err": sample["feh_e"],
		# "[mg/fe]_err": len(sample["feh"]) * [0.01],
		"[mg/fe]_err": sample["mgfe_e"],
		"lookback": [m.log10(age) for age in sample["age"]],
		"lookback_err": [m.log10((upper - lower) / 2) for upper, lower in zip(
			sample["age_ep"], sample["age_em"])]
		# "logage": [m.log10(age) for age in sample["age"]],
		# "logage_err": [m.log10(upper - lower) for upper, lower in zip(
		# 	sample["age_ep"], sample["age_em"])]
		})
	pool = Pool(N_PROC)
	sampler = EnsembleSampler(N_WALKERS, N_DIM, logl, pool = pool)
	# sampler = EnsembleSampler(N_WALKERS, N_DIM, logl)
	p0 = 10 * np.random.rand(N_WALKERS, N_DIM)
	for i in range(len(p0)):
		p0[i][4] /= 1000
		p0[i][5] /= 1000
	start = time.time()
	state = sampler.run_mcmc(p0, N_BURNIN)
	sampler.reset()
	state = sampler.run_mcmc(state, N_ITERS)
	stop = time.time()
	print("MCMC time: ", stop - start)
	savechain(sampler, OUTFILE)




