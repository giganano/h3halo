
from multiprocessing import Pool
from emcee import EnsembleSampler
from src import mcmc
from src.utils import linear_exponential, savechain, piecewise_linear
from src.utils import gaussian, double_gaussian
import numpy as np
import math as m
import numbers
import random
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
N_TIMESTEPS = 500
N_WALKERS = 256
N_BURNIN = 1000
N_ITERS = 2000
COSMOLOGICAL_AGE = 13.2
N_DIM = 9

# emcee walker parameters (double gaussian SFH plus gaussian SFE-driven burst)
#
# 0. center of first SF event
# 1. width of first SF event
# 2. amplitude ratio of second to first SF events
# 3. center of second SF event
# 4. width of second SF event
# 5. mass loading factor
# 6. maximum SFE timescale
# 7. fractional tau_star decrease
# 8. total duration of the model


# emcee walker parameters (pure SFE burst)
#
# 0. infall timescale
# 1. mass loading factor
# 2. initial SFE timescale
# 3. final SFE timescale
# 4. onset of SFE rampup
# 5. duration of SFE rampup
# 6. total duration of the model
### 7. IMF-averaged Fe yield from CCSNe
### 8. DTD-integrated Fe yield from SNe Ia


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


class gaussian_SFE_burst(gaussian):

	def __init__(self, constant = 0, **kwargs):
		super().__init__(**kwargs)
		self.constant = constant

	def __call__(self, x):
		return self._constant * (1 - super().__call__(x))

	@property
	def constant(self):
		return self._constant

	@constant.setter
	def constant(self, value):
		if isinstance(value, numbers.Number):
			if value >= 0:
				self._constant = float(value)
			else:
				raise ValueError("Constant must be non-negative.")
		else:
			raise TypeError("Constant must be a real number. Got: %s" % (
				type(value)))


class sgrfit(mcmc):

	def __init__(self, data):
		super().__init__(data)
		self.sz.elements = ["fe", "o"]

		# double gaussian SFH with gaussian SFE burst
		self.sz.func = double_gaussian()
		self.sz.func.first.amplitude = 100
		self.sz.mode = "sfr"
		self.sz.tau_star = gaussian_SFE_burst()

		# linear exponential IFR with SFE-driven burst
		# self.sz.func = linear_exponential()
		# self.sz.mode = "ifr"
		# self.sz.tau_star = rampup(100, 10, 5, 1)
		# self.sz.Mg0 = 0

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		if walker[8] > COSMOLOGICAL_AGE: return -float("inf")
		# negative tau_star during burst
		if walker[7] > 1: return -float("inf")
		# if walker[2] < walker[3]: return -float("inf")
		# if walker[4] > walker[6]: return -float("inf")
		print(np.array(walker))
		self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())

		# double gaussian SFH with gaussian SFE-burst
		# self.sz.func.first.amplitude = walker[0]
		self.sz.func.first.mean = walker[0]
		self.sz.func.first.width = walker[1]
		self.sz.func.second.amplitude = 100 * walker[2]
		self.sz.func.second.mean = walker[3]
		self.sz.func.second.width = walker[4]
		self.sz.eta = walker[5]
		self.sz.tau_star.constant = walker[6]
		self.sz.tau_star.amplitude = walker[7]
		self.sz.tau_star.mean = walker[3]
		self.sz.tau_star.width = walker[4]
		# self.sz.tau_star.constant = walker[6]
		# self.sz.tau_star.amplitude = walker[7]
		# self.sz.tau_star.mean = walker[8]
		# self.sz.tau_star.width = walker[9]
		self.sz.dt = walker[8] / N_TIMESTEPS
		output = self.sz.run(np.linspace(0, walker[8], N_TIMESTEPS + 1),
			overwrite = True, capture = True)

		# linear-exponential IFR with SFE-driven burst
		# self.sz.func.timescale = walker[0]
		# self.sz.eta = walker[1]
		# self.sz.tau_star.initial = walker[2]
		# self.sz.tau_star.final = walker[3]
		# self.sz.tau_star.onset = walker[4]
		# self.sz.tau_star.rampup = walker[5]
		# self.sz.dt = walker[6] / N_TIMESTEPS
		# output = self.sz.run(np.linspace(0, walker[6], N_TIMESTEPS + 1),
		# 	overwrite = True, capture = True)

		model = []
		for key in self.quantities:
			model.append(output.history[key][1:])
		model = np.array(model).T
		self.fd.model = model
		self.fd.weights = output.history["sfr"][1:]
		return self.fd()

if __name__ == "__main__":
	raw = np.genfromtxt(DATA_FILE)
	data = vice.dataframe({
		"[fe/h]": np.array([row[0] for row in raw]),
		"[fe/h]_err": np.array([row[1] for row in raw]),
		"[o/fe]": np.array([row[2] for row in raw]),
		"[o/fe]_err": np.array([row[3] for row in raw])
	})
	data = data.filter("[o/fe]", "<=", 0.35).todict()
	log_prob = sgrfit(data)
	pool = Pool(N_PROC)
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob, pool = pool)
	p0 = np.zeros((N_WALKERS, N_DIM))
	# random.seed(a = 0)
	for i in range(len(p0)):
		# double gaussian SFH
		p0[i][0] = 11
		p0[i][1] = 3
		p0[i][2] = 4
		p0[i][3] = 6
		p0[i][4] = 2
		p0[i][5] = 25
		p0[i][6] = 100
		p0[i][7] = 0.8
		p0[i][8] = 10

		# linear-exponential IFR with SFE-driven burst
		# p0[i][0] = 2
		# p0[i][1] = 25
		# p0[i][2] = 100
		# p0[i][3] = 10
		# p0[i][4] = 6
		# p0[i][5] = 1
		# p0[i][6] = 8
		for j in range(len(p0[i])):
			p0[i][j] += np.random.normal(scale = 0.3 * p0[i][j])
	start = time.time()
	state = sampler.run_mcmc(p0, N_BURNIN)
	sampler.reset()
	state = sampler.run_mcmc(state, N_ITERS)
	stop = time.time()
	print("MCMC time: ", stop - start)
	savechain(sampler, OUTFILE)

