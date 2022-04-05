r"""
ARGV
----
1) The name of the file containing the mock sample
2) The name of the output file
3) The name of the VICE output to store each iterations output in
4) The number of walkers to use
5) The number of iterations to run each walker through
6) The number of timesteps to use in each one-zone model integration
7) The number of burn-in steps to run
8) The number of cores to spread the calculation across
"""

from multiprocessing import Pool
from emcee import EnsembleSampler
from src import mcmc
from src.utils import exponential, savechain
import numpy as np
import math as m
import vice
from vice.yields.presets import JW20
import time
import sys
import os

ENDTIME = 10
N_TIMESTEPS = int(sys.argv[6])
N_WALKERS = int(sys.argv[4])
N_DIM = 3


class expifr_mcmc(mcmc):

	def __init__(self, data):
		super().__init__(data)
		self.sz.elements = ["fe", "o"]
		self.sz.func = exponential()
		self.sz.mode = "ifr"
		self.sz.Mg0 = 0
		self.sz.nthreads = 2
		self.sz.dt = ENDTIME / N_TIMESTEPS

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		self.sz.name = "%s%s" % (sys.argv[3], os.getpid())
		self.sz.func.timescale = walker[0]
		self.sz.tau_star = walker[1]
		self.sz.eta = walker[2]
		result = super().__call__(self.sz.run(
			np.linspace(0, ENDTIME, N_TIMESTEPS + 1),
			overwrite = True, capture = True))
		print("walker: [%.5e, %.5e, %.5e] " % (walker[0], walker[1], walker[2]))
		print(result)
		return result


if __name__ == "__main__":
	raw = np.genfromtxt(sys.argv[1])
	data = {
		"[fe/h]": np.array([row[0] for row in raw]),
		"[fe/h]_err": np.array([row[1] for row in raw]),
		"[o/fe]": np.array([row[2] for row in raw]),
		"[o/fe]_err": np.array([row[3] for row in raw]),
		"lookback": np.array([row[4] for row in raw]),
		"lookback_err": np.array([row[5] for row in raw])
	}
	log_prob = expifr_mcmc(data)
	pool = Pool(int(sys.argv[8]))
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob, pool = pool)
	# start initial at known position anyway since this is a mock
	p0 = N_WALKERS * [None]
	for i in range(len(p0)):
		p0[i] = [2, 10, 25]
		for j in range(len(p0[i])):
			p0[i][j] += np.random.normal(scale = 0.1 * p0[i][j])
	p0 = np.array(p0)
	start = time.time()
	state = sampler.run_mcmc(p0, int(sys.argv[7]))
	sampler.reset()
	state = sampler.run_mcmc(state, int(sys.argv[5]))
	stop = time.time()
	print("MCMC time: ", stop - start)
	savechain(sampler, sys.argv[2])

