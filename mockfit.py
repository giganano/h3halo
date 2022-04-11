
from multiprocessing import Pool
from emcee import EnsembleSampler
from src import mcmc
from src.utils import exponential, savechain, piecewise_linear
import numpy as np
import math as m
import vice
from vice.yields.presets import JW20
vice.yields.ccsne.settings['mg'] = 0.00261
vice.yields.sneia.settings['mg'] = 0
import time
import sys
import os

DATA_FILE = "./mocksamples/simpleburst.dat"
OUTFILE = "./mocksamples/simpleburst_noages_5000.out"
MODEL_BASENAME = "simpleburst"
N_PROC = 10
N_TIMESTEPS = 1000
N_WALKERS = 50
N_BURNIN = 100
N_ITERS = 100
H3_UNIVERSE_AGE = 14
N_DIM = 7

# emcee walker parameters
#
# 0. infall timescale
# 1. Mass loading factor
# 2. total duration of the model
# 3. value of initial tau_star
# 4. duration of initial tau_star
# 5. duration of tau_star decrease
# 6. value of final tau_star


class expifr_mcmc(mcmc):

	def __init__(self, data):
		super().__init__(data)
		self.sz.elements = ["fe", "o"]
		self.sz.func = exponential()
		self.sz.mode = "ifr"
		self.sz.Mg0 = 0
		self.sz.nthreads = 2
		self.sz.tau_star = piecewise_linear(2)
		self.sz.tau_star.slopes[0] = 0
		self.sz.tau_star.slopes[2] = 0

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		if walker[2] > H3_UNIVERSE_AGE: return -float("inf")
		print("walker: [%.2f, %.2f, %.2f, %.2e, %.2f, %.2f, %.2f] " % (
			walker[0], walker[1], walker[2], walker[3],
			walker[4], walker[5], walker[6]))
		self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())
		self.sz.func.timescale = walker[0]
		# self.sz.tau_star = walker[1]
		self.sz.eta = walker[1]
		self.sz.dt = walker[2] / N_TIMESTEPS
		self.sz.tau_star.norm = walker[3]
		self.sz.tau_star.deltas[0] = walker[4]
		self.sz.tau_star.deltas[1] = walker[5]
		self.sz.tau_star.slopes[1] = (walker[6] - walker[3]) / walker[5]
		output = self.sz.run(np.linspace(0, walker[2], N_TIMESTEPS + 1),
			overwrite = True, capture = True)
		diff = H3_UNIVERSE_AGE - walker[2]
		model = []
		for key in self.quantities:
			if key == "lookback":
				model.append(
					[m.log10(_ + diff) for _ in output.history[key][:-1]])
			else:
				model.append(output.history[key][1:])
		model = np.array(model).T
		weights = output.history["sfr"][1:]
		norm = sum(weights)
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
	log_prob = expifr_mcmc(data)
	pool = Pool(int(N_PROC))
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob, pool = pool)
	# start initial at known position anyway since this is a mock
	p0 = N_WALKERS * [None]
	for i in range(len(p0)):
		# p0[i] = [2, 10, 25, 10]
		p0[i] = [2, 10, 5, 50, 2.5, 1, 2]
		for j in range(len(p0[i])):
			p0[i][j] += np.random.normal(scale = 0.1 * p0[i][j])
	p0 = np.array(p0)
	start = time.time()
	state = sampler.run_mcmc(p0, N_BURNIN)
	sampler.reset()
	state = sampler.run_mcmc(state, N_ITERS)
	stop = time.time()
	print("MCMC time: ", stop - start)
	savechain(sampler, OUTFILE)

