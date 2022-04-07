
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
import sys
import os

DATA_FILE = "./mocksamples/someages_offset4.dat"
OUTFILE = "./mocksamples/someages_offset4_wyields_5000.out"
MODEL_BASENAME = "someages_wyields"
N_PROC = 4
N_TIMESTEPS = 1000
N_WALKERS = 50
N_BURNIN = 100
N_ITERS = 100
H3_UNIVERSE_AGE = 14
N_DIM = 7

# emcee walker parameters
#
# 1. infall timescale
# 2. SFE timescale
# 3. Mass loading factor
# 4. total duration of the model
# 5. CCSN O yield
# 6. CCSN Fe yield
# 7. SN Ia Fe yield


class expifr_mcmc(mcmc):

	def __init__(self, data):
		super().__init__(data)
		self.sz.elements = ["fe", "o"]
		self.sz.func = exponential()
		self.sz.mode = "ifr"
		self.sz.Mg0 = 0
		self.sz.nthreads = 2
		# self.sz.dt = ENDTIME / N_TIMESTEPS

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		if walker[3] > H3_UNIVERSE_AGE: return -float("inf")
		if not 0.003 <= walker[4] <= 0.075: return -float("inf")
		if not 0.00024 <= walker[5] <= 0.006: return -float("inf")
		if not 0.00034 <= walker[6] <= 0.0085: return -float("inf")
		print("walker: [%.2f, %.2f, %.2f, %.2f, %.2e, %.2e, %.2e] " % (
			walker[0], walker[1], walker[2], walker[3], walker[4], walker[5],
			walker[6]))
		vice.yields.ccsne.settings['o'] = walker[4]
		vice.yields.ccsne.settings['fe'] = walker[5]
		vice.yields.sneia.settings['fe'] = walker[6]
		self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())
		self.sz.func.timescale = walker[0]
		self.sz.tau_star = walker[1]
		self.sz.eta = walker[2]
		self.sz.dt = walker[3] / N_TIMESTEPS
		output = self.sz.run(np.linspace(0, walker[3], N_TIMESTEPS + 1),
			overwrite = True, capture = True)
		diff = H3_UNIVERSE_AGE - walker[3]
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
		weights = [_ / norm for _ in weights]
		self.fd.model = model
		self.fd.weights = weights
		return self.fd()
		# return super().__call__(self.sz.run(
		# 	np.linspace(0, walker[3], N_TIMESTEPS + 1),
		# 	overwrite = True, capture = True))


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
		# p0[i] = [2, 10, 25]
		p0[i] = [2, 10, 25, 10, 0.015, 0.0012, 0.0017]
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

