
from multiprocessing import Pool
from emcee import EnsembleSampler
from src import mcmc
from src.utils import linear_exponential, savechain, piecewise_linear
import numpy as np
import math as m
import vice
vice.yields.ccsne.settings['o'] = 0.01
vice.yields.sneia.settings['o'] = 0
vice.yields.ccsne.settings['fe'] = 0.0008
vice.yields.sneia.settings['fe'] = 0.0011
import time
import sys
import os

DATA_FILE = sys.argv[1]
OUTFILE = sys.argv[2]
MODEL_BASENAME = sys.argv[3]


N_PROC = 40
N_TIMESTEPS = 1000
N_WALKERS = 256
N_BURNIN = 200
N_ITERS = 100
COSMOLOGICAL_AGE = 13.2
N_DIM = 9

# emcee walker parameters
#
# 0. infall timescale
# 1. mass loading factor
# 2. initial SFE timescale
# 3. final SFE timescale
# 4. onset of SFE rampup
# 5. duration of SFE rampup
# 6. total duration of the model
# 7. IMF-averaged Fe yield from CCSNe
# 8. DTD-integrated Fe yield from SNe Ia


class rampup:

	def __init__(self, initial, final, onset, rampup):
		self.initial = initial
		self.final = final
		self.onset = onset
		self.rampup = rampup

	def __call__(self, time):
		if time < self.onset:
			return self.initial
		elif self.onset <= time <= self.onset + self.rampup:
			return self.initial - (self.initial -
				self.final) / self.rampup * (time - self.onset)
		else:
			return self.final


class sgrfit(mcmc):

	def __init__(self, data):
		super().__init__(data)
		self.sz.elements = ["fe", "o"]
		self.sz.func = linear_exponential()
		self.sz.mode = "ifr"
		self.sz.tau_star = rampup(100, 10, 5, 1)
		self.sz.Mg0 = 0

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		if walker[6] > COSMOLOGICAL_AGE: return -float("inf")
		if walker[2] < walker[3]: return -float("inf")
		if walker[4] > walker[6]: return -float("inf")
		print("""\
walker: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2e, %.2e]""" % (
			walker[0], walker[1], walker[2], walker[3], walker[4], walker[5],
			walker[6], walker[7], walker[8]))
		self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())
		self.sz.func.timescale = walker[0]
		self.sz.eta = walker[1]
		self.sz.tau_star.initial = walker[2]
		self.sz.tau_star.final = walker[3]
		self.sz.tau_star.onset = walker[4]
		self.sz.tau_star.rampup = walker[5]
		self.sz.dt = walker[6] / N_TIMESTEPS
		vice.yields.ccsne.settings['fe'] = walker[7]
		vice.yields.sneia.settings['fe'] = walker[8]
		output = self.sz.run(np.linspace(0, walker[6], N_TIMESTEPS + 1),
			overwrite = True, capture = True)
		model = []
		for key in self.quantities:
			model.append(output.history[key][1:])
		model = np.array(model).T
		self.fd.model = model
		self.fd.weights = output.history["sfr"][1:]
		return self.fd()

if __name__ == "__main__":
	raw = np.genfromtxt(DATA_FILE)
	data = {
		"[fe/h]": np.array([row[0] for row in raw]),
		"[fe/h]_err": np.array([row[1] for row in raw]),
		"[o/fe]": np.array([row[2] for row in raw]),
		"[o/fe]_err": np.array([row[3] for row in raw])
	}
	log_prob = sgrfit(data)
	pool = Pool(N_PROC)
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob, pool = pool)
	p0 = 1000 * np.random.rand(N_WALKERS, N_DIM)
	for i in range(len(p0)):
		p0[i][7] /= 100000
		p0[i][8] /= 100000
	start = time.time()
	state = sampler.run_mcmc(p0, N_BURNIN)
	sampler.reset()
	state = sampler.run_mcmc(state, N_ITERS)
	stop = time.time()
	print("MCMC time: ", stop - start)
	savechain(sampler, OUTFILE)

