
from multiprocessing import Pool
from emcee import EnsembleSampler
from src import mcmc
from src.utils import exponential, savechain, sinusoid
import numpy as np
import math as m
import vice
from vice.yields.presets import JW20
vice.yields.ccsne.settings['mg'] = 0.00261
vice.yields.sneia.settings['mg'] = 0
import time
import sys
import os

DATA_FILE = "./mocksamples/sinusoidal.dat"
OUTFILE = "./mocksamples/sinusoidal_5000.out"
MODEL_BASENAME = "sinusoidal"
N_PROC = 10
N_TIMESTEPS = 1000
N_WALKERS = 50
N_BURNIN = 100
N_ITERS = 100
H3_UNIVERSE_AGE = 14
N_DIM = 6

# emcee walker parameters
#
# 0. fractional amplitude of infall sinusoid
# 1. period of infall sinusoid
# 2. phase shift of infall sinusoid
# 3. mass loading factor
# 4. tau_star
# 5. total duration of the model

class sinusoid_mcmc(mcmc):

	def __init__(self, data):
		super().__init__(data)
		self.sz.elements = ["fe", "o"]
		self.sz.func = sinusoid(mean = 1)
		self.sz.mode = "ifr"
		self.sz.Mg0 = 0
		self.sz.nthreads = 2

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		if walker[5] > H3_UNIVERSE_AGE: return -float("inf")
		if walker[0] > 1: return -float("inf")
		if walker[2] > walker[1]: return -float("inf")
		print("walker: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]" % (
			walker[0], walker[1], walker[2], walker[3], walker[4],
			walker[5]))
		self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())
		self.sz.func.amplitude = walker[0]
		self.sz.func.period = walker[1]
		self.sz.func.shift = walker[2]
		self.sz.eta = walker[3]
		self.sz.tau_star = walker[4]
		self.sz.dt = walker[5] / N_TIMESTEPS
		output = self.sz.run(np.linspace(0, walker[5], N_TIMESTEPS + 1),
			overwrite = True, capture = True)
		diff = H3_UNIVERSE_AGE - walker[5]
		model = []
		for key in self.quantities:
			if key == "lookback":
				model.append(
					[m.log10(_ + diff) for _ in output.history[key][:-1]])
			else:
				model.append(output.history[key][1:])
		model = np.array(model).T
		dt = output.history["time"][1] - output.history["time"][0]
		weights = [_ * dt for _ in output.history["sfr"][1:]]
		self.fd.model = model
		self.fd.weights = weights
		return self.fd()


if __name__ == "__main__":
	raw = np.genfromtxt(DATA_FILE)
	data = {
		"[fe/h]": np.array([row[0] for row in raw]),
		"[fe/h]_err": np.array([row[1] for row in raw]),
		"[o/fe]": np.array([row[2] for row in raw]),
		"[o/fe]_err": np.array([row[3] for row in raw]),
		"lookback": np.array([row[4] for row in raw]),
		"lookback_err": np.array([row[5] for row in raw])
	}
	log_prob = sinusoid_mcmc(data)
	pool = Pool(int(N_PROC))
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob, pool = pool)
	p0 = N_WALKERS * [None]
	for i in range(len(p0)):
		p0[i] = [0.4, 2, 0.5, 15, 10, 10]
		for j in range(len(p0[i])):
			p0[i][j] += np.random.normal(0.1 * p0[i][j])
	p0 = np.array(p0)
	start = time.time()
	state = sampler.run_mcmc(p0, N_BURNIN)
	sampler.reset()
	state = sampler.run_mcmc(state, N_ITERS)
	stop = time.time()
	print("MCMC time: ", stop - start)
	savechain(sampler, OUTFILE)




