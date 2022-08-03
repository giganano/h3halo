
from multiprocessing import Pool
from emcee import EnsembleSampler
from src import mcmc
from src.utils import exponential, savechain, linear_exponential
import numpy as np
import math as m
import numbers
import vice
from vice.yields.presets import JW20
vice.yields.ccsne.settings['o'] = 0.01
vice.yields.sneia.settings['o'] = 0
import time
import os

DATA_FILE = "./data/wukong/wukong.dat"
OUTFILE = "./data/wukong/wukong_102k4.out"
MODEL_BASENAME = "wukongfit"


N_PROC = 40
N_TIMESTEPS = 500
N_WALKERS = 256
N_BURNIN = 200
N_ITERS = 400
COSMOLOGICAL_AGE = 13.2
N_DIM = 6

# emcee walker parameters (exponential IFR)
#
# 0. infall timescale
# 1. mass loading factor
# 2. SFE timescale
# 3. total duration of the model
# 4. IMF-averaged Fe yield from CCSNe
# 5. DTD-integrated Fe yield from SNe Ia


class wukongfit(mcmc):

	def __init__(self, data):
		super().__init__(data)
		self.sz.elements = ["fe", "o"]
		self.sz.func = exponential(prefactor = 1000)
		self.sz.mode = "ifr"
		self.sz.Mg0 = 0

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		if walker[3] > COSMOLOGICAL_AGE: return -float("inf")
		print("walker: [%.2f, %.2f, %.2f, %.2f, %.2e, %.2e]" % (walker[0],
			walker[1], walker[2], walker[3], walker[4], walker[5]))
		self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())
		self.sz.func.timescale = walker[0]
		self.sz.eta = walker[1]
		self.sz.tau_star = walker[2]
		self.sz.dt = walker[3] / N_TIMESTEPS
		vice.yields.ccsne.settings['fe'] = walker[4]
		vice.yields.sneia.settings['fe'] = walker[5]
		output = self.sz.run(np.linspace(0, walker[3], N_TIMESTEPS + 1),
			overwrite = True, capture = True)
		diff = COSMOLOGICAL_AGE - walker[3]
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
		"[o/fe]_err": np.array([row[3] for row in raw]),
	}
	log_prob = wukongfit(data)
	pool = Pool(N_PROC)
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob, pool = pool)
	p0 = 10 * np.random.rand(N_WALKERS, N_DIM)
	# confine the [a/Fe] plateau to the allowed range to begin with
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

