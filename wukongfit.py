
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
vice.yields.ccsne.settings['fe'] = 7.78e-4
vice.yields.sneia.settings['fe'] = 1.23e-3
import time
import os

DATA_FILE = "./data/wukong/wukong.dat"
OUTFILE = "./data/wukong/wukong_constantsfr_512k.out"
MODEL_BASENAME = "wukongfit"


N_PROC = 40
N_TIMESTEPS = 500
N_WALKERS = 256
N_BURNIN = 1000
N_ITERS = 2000
COSMOLOGICAL_AGE = 13.2
N_DIM = 3

# emcee walker parameters (exponential IFR)
#
# 0. infall timescale
# 1. mass loading factor
# 2. SFE timescale
# 3. total duration of the model
# 4. IMF-averaged Fe yield from CCSNe
# 5. DTD-integrated Fe yield from SNe Ia

# emcee walker parameters (constant SFR)
#
# 0. mass loading factor
# 1. SFE timescale
# 2. total duration of the model
### 3. IMF-averaged Fe yield from CCSNe
### 4. DTD-integrated Fe yield from SNe Ia


def constantsfr(t):
	return 1000


class wukongfit(mcmc):

	def __init__(self, data):
		super().__init__(data)
		self.sz.elements = ["fe", "o"]
		# self.sz.func = exponential(prefactor = 1000)
		# self.sz.mode = "ifr"
		self.sz.func = constantsfr
		self.sz.mode = "sfr"
		self.sz.Mg0 = 0

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		if walker[2] > COSMOLOGICAL_AGE: return -float("inf")
		print("walker: [%.2f, %.2f, %.2f]" % (walker[0], walker[1], walker[2]))
		self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())
		# self.sz.func.timescale = walker[0]
		self.sz.eta = walker[0]
		self.sz.tau_star = walker[1]
		self.sz.dt = walker[2] / N_TIMESTEPS
		# vice.yields.ccsne.settings['fe'] = walker[3]
		# vice.yields.sneia.settings['fe'] = walker[4]
		output = self.sz.run(np.linspace(0, walker[2], N_TIMESTEPS + 1),
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
		"[o/fe]_err": np.array([row[3] for row in raw]),
	}
	log_prob = wukongfit(data)
	pool = Pool(N_PROC)
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob, pool = pool)
	p0 = 10 * np.random.rand(N_WALKERS, N_DIM)
	# confine the [a/Fe] plateau to the allowed range to begin with
	# for i in range(len(p0)):
	# 	p0[i][3] /= 1000
	# 	p0[i][4] /= 1000
	start = time.time()
	state = sampler.run_mcmc(p0, N_BURNIN)
	sampler.reset()
	state = sampler.run_mcmc(state, N_ITERS)
	stop = time.time()
	print("MCMC time: ", stop - start)
	savechain(sampler, OUTFILE)

