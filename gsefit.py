
from multiprocessing import Pool
from emcee import EnsembleSampler
from src import mcmc
from src.utils import exponential, savechain
import numpy as np
import math as m
import vice
from vice.yields.presets import JW20
vice.yields.ccsne.settings['mg'] = 0.00261
vice.yields.sneia.settings['mg'] = 0
import time
import os

DATA_FILE = "./data/gsechem.dat"
OUTFILE = "./data/gsechem.out"
MODEL_BASENAME = "gsefit"
N_PROC = 10
N_TIMESTEPS = 1000
N_WALKERS = 256
N_BURNIN = 100
N_ITERS = 400
H3_UNIVERSE_AGE = 14
N_DIM = 4


class gsefit(mcmc):

	def __init__(self, data):
		super().__init__(data)
		# self.sz.elements = ["fe", "o"]
		self.sz.elements = ["fe", "mg"]
		self.sz.func = exponential()
		self.sz.mode = "ifr"
		self.sz.Mg0 = 0
		self.sz.nthreads = 2

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		if walker[3] > H3_UNIVERSE_AGE: return -float("inf")
		print("walker: [%.2f, %.2f, %.2f, %.2f]" % (walker[0], walker[1],
			walker[2], walker[3]))
			# walker[2]))
		self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())
		self.sz.dt = walker[3] / N_TIMESTEPS
		self.sz.func.timescale = walker[0]
		self.sz.tau_star = walker[1]
		self.sz.eta = walker[2]
		output = self.sz.run(np.linspace(0, walker[3], N_TIMESTEPS + 1),
			overwrite = True, capture = True)
		# output = self.sz.run(np.linspace(0, 10, N_TIMESTEPS + 1),
		# 	overwrite = True, capture = True)
		diff = H3_UNIVERSE_AGE - walker[3]
		# diff = 0
		model = []
		for key in self.quantities:
			if key == "lookback":
				model.append(
					# take into account offset between GSE stopping evolution
					# and the present day
					[m.log10(_ + diff) for _ in output.history[key][:-1]])
			else:
				model.append(output.history[key][1:])
		model = np.array(model).T
		weights = output.history["sfr"][1:]
		norm = sum(weights)
		weights = [_ / norm for _ in weights]
		self.fd.model = model
		self.fd.weights = weights
		logp = self.fd()
		print(logp)
		return logp


if __name__ == "__main__":
	raw = np.genfromtxt(DATA_FILE)
	data = {
		"[fe/h]": np.array([row[0] for row in raw]),
		"[fe/h]_err": np.array([row[1] for row in raw]),
		"[mg/fe]": np.array([row[2] for row in raw]),
		"[mg/fe]_err": np.array([row[3] for row in raw]),
		"lookback": np.array([row[4] for row in raw]),
		"lookback_err": np.array([row[5] for row in raw])
	}
	log_prob = gsefit(data)
	pool = Pool(N_PROC)
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob, pool = pool)
	p0 = 10 * np.random.rand(N_WALKERS, N_DIM)
	start = time.time()
	state = sampler.run_mcmc(p0, N_BURNIN)
	sampler.reset()
	state = sampler.run_mcmc(state, N_ITERS)
	stop = time.time()
	print("MCMC time: ", stop - start)
	savechain(sampler, OUTFILE)

