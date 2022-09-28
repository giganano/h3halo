
from multiprocessing import Pool
from emcee import EnsembleSampler
from src import mcmc
from src.utils import exponential, savechain, piecewise_linear
import numpy as np
import math as m
import vice
# from vice.yields.presets import JW20
vice.yields.ccsne.settings['o'] = 0.01
vice.yields.sneia.settings['o'] = 0
vice.yields.ccsne.settings['fe'] = 0.0008
vice.yields.sneia.settings['fe'] = 0.0011
import time
import sys
import os

# DATA_FILE = "./mocksamples/fiducial.dat"
# OUTFILE = "./mocksamples/fiducial_5000.out"
# MODEL_BASENAME = "fiducial"
# DATA_FILE = "./mocksamples/n20.dat"
# OUTFILE = "./mocksamples/n20_5000.out"
# MODEL_BASENAME = "n20_"
DATA_FILE = sys.argv[1]
OUTFILE = sys.argv[2]
MODEL_BASENAME = sys.argv[3]


N_PROC = 40
N_TIMESTEPS = 500
N_WALKERS = 256
N_BURNIN = 1000
N_ITERS = 2000
COSMOLOGICAL_AGE = 13.2
N_DIM = 6

# emcee walker parameters
#
# 0. infall timescale
# 1. mass loading factor
# 2. SFE timescale
# 3. total duration of the model
# 4. IMF-averaged Fe yield from CCSNe
# 5. DTD-integrated Fe yield from SNe Ia
# 6 (temporary). IMF-averaged alpha-element yield


class expifr_mcmc(mcmc):

	def __init__(self, data):
		super().__init__(data)
		self.sz.elements = ["fe", "o"]
		self.sz.func = exponential()
		self.sz.mode = "ifr"
		self.sz.Mg0 = 0
		# self.sz.nthreads = 2

	def __call__(self, walker):
		# strict bound because of physics
		if any([_ < 0 for _ in walker]): return -float("inf")
		if walker[3] > COSMOLOGICAL_AGE: return -float("inf")
		if walker[4] > 0.1: return -float("inf")
		if walker[5] > 0.1: return -float("inf")
		# if walker[6] > 0.1: return -float("inf")
		print("walker: [%.2f, %.2f, %.2f, %.2f, %.2e, %.2e]" % (
			walker[0], walker[1], walker[2], walker[3], walker[4], walker[5]))
		self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())
		self.sz.func.timescale = walker[0]
		self.sz.eta = walker[1]
		self.sz.tau_star = walker[2]
		self.sz.dt = walker[3] / N_TIMESTEPS
		vice.yields.ccsne.settings['fe'] = walker[4]
		vice.yields.sneia.settings['fe'] = walker[5]
		# vice.yields.ccsne.settings['o'] = walker[6]
		output = self.sz.run(np.linspace(0, walker[3], N_TIMESTEPS + 1), 
			overwrite = True, capture = True)
		diff = COSMOLOGICAL_AGE - walker[3]
		model = []
		for key in self.quantities:
			if key == "lookback":
				model.append(
					[m.log10(_ + diff) for _ in output.history[key][:-1]])
			else:
				model.append(output.history[key][1:])
		model = np.array(model).T
		self.fd.model = model
		self.fd.weights = output.history["sfr"][1:]
		return self.fd()

		# print("walker: [%.2f, %.2f, %.2f, %.2f]" % (
		# 	walker[0], walker[1], walker[2], walker[3]))
		# self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())
		# self.sz.func.timescale = walker[0]
		# self.sz.eta = walker[1]
		# self.sz.tau_star = walker[2]
		# self.sz.dt = walker[3] / N_TIMESTEPS
		# output = self.sz.run(np.linspace(0, walker[3], N_TIMESTEPS + 1),
		# 	overwrite = True, capture = True)
		# diff = H3_UNIVERSE_AGE - walker[3]
		# model = []
		# for key in self.quantities:
		# 	if key == "lookback":
		# 		model.append(
		# 			[m.log10(_ + diff) for _ in output.history[key][:-1]])
		# 	else:
		# 		model.append(output.history[key][1:])
		# model = np.array(model).T
		# self.fd.model = model
		# self.fd.weights = output.history["sfr"][1:]
		# return self.fd()


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
		p0[i] = [2, 10, 15, 10, 0.0008, 0.0011]
		# p0[i] = [2, 10, 15, 10, 0.0008, 0.0011, 0.01]
		for j in range(len(p0[i])):
			p0[i][j] += np.random.normal(scale = 0.1 * p0[i][j])
	p0 = np.array(p0)
	start = time.time()
	state = sampler.run_mcmc(p0, N_BURNIN)
	sampler.reset()
	state = sampler.run_mcmc(state, N_ITERS)
	stop = time.time()
	pool.close()
	pool.join()
	print("MCMC time: ", stop - start)
	savechain(sampler, OUTFILE)

