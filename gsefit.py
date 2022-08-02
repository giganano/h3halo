
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
# vice.yields.ccsne.settings['mg'] = 0.00261
# vice.yields.sneia.settings['mg'] = 0
import time
import os

DATA_FILE = "./data/gse/gsechem.dat"
OUTFILE = "./data/gse/gsechem_512k.out"
MODEL_BASENAME = "gsefit"


N_PROC = 40
N_TIMESTEPS = 500
N_WALKERS = 256
N_BURNIN = 1000
N_ITERS = 2000
COSMOLOGICAL_AGE = 13.2
N_DIM = 6

# emcee walker parameters (constant-then-exponential IFR)
#
# 0. Duration of constant infall
# 1. E-folding timescale of exponential infall phase
# 2. mass loading factor
# 3. SFE timescale
# 4. total duration of the model
# 5. IMF-averaged Fe yield from CCSNe
# 6. DTD-integrated Fe yield from SNe Ia


# emcee walker parameters (linear-exponential SFH)
#
# 0. SFH timescale
# 1. mass loading factor
# 2. SFE timescale
# 3. total duration of the model
# 4. IMF-averaged Fe yield from CCSNe
# 5. DTD-integrated Fe yield from SNe Ia

# emcee walker parameters ((linear-)exponential IFR)
#
# 0. infall timescale
# 1. mass loading factor
# 2. SFE timescale
# 3. total duration of the model
# 4. IMF-averaged Fe yield from CCSNe
# 5. DTD-integrated Fe yield from SNe Ia

# class constant_then_exponential(exponential):

# 	def __init__(self, onset = 0, **kwargs):
# 		super().__init__(**kwargs)
# 		self.onset = onset

# 	def __call__(self, time):
# 		if time < self._onset:
# 			return self._prefactor
# 		else:
# 			return super().__call__(time - self._onset)

# 	@property
# 	def onset(self):
# 		return self._onset

# 	@onset.setter
# 	def onset(self, value):
# 		if isinstance(value, numbers.Number):
# 			if value >= 0:
# 				self._onset = float(value)
# 			else:
# 				raise ValueError("Onset must be positive.")
# 		else:
# 			raise TypeError("Onset must be a real number.")


class gsefit(mcmc):

	def __init__(self, data):
		super().__init__(data)
		self.sz.elements = ["fe", "o"]
		# self.sz.func = linear_exponential(prefactor = 1000)
		# self.sz.mode = "sfr"
		# self.sz.func = constant_then_exponential(prefactor = 1000)
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
		# self.sz.func.onset = walker[0]
		# self.sz.func.timescale = walker[1]
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
			if key == "lookback":
				model.append(
					[m.log10(_ + diff) for _ in output.history[key][:-1]])
			else:
				model.append(output.history[key][1:])
		model = np.array(model).T
		self.fd.model = model
		self.fd.weights = output.history["sfr"][1:]
		return self.fd()

		# if any([_ < 0 for _ in walker]): return -float("inf")
		# if walker[3] > H3_UNIVERSE_AGE: return -float("inf")
		# if walker[4] < 0.1 or walker[4] > 0.8: return -float("inf")
		# print("walker: [%.2f, %.2f, %.2f, %.2f, %.2f]" % (walker[0], walker[1],
		# 	walker[2], walker[3], walker[4]))
		# self.sz.name = "%s%s" % (MODEL_BASENAME, os.getpid())
		# self.sz.dt = walker[3] / N_TIMESTEPS
		# self.sz.func.timescale = walker[0]
		# self.sz.tau_star = walker[1]
		# self.sz.eta = walker[2]

		# # equilibrium [a/Fe] ratio for constant SFR given our fiducial yields
		# afe_eq = 0.067
		# vice.yields.ccsne.settings["fe"] = vice.yields.ccsne.settings["o"] * (
		# 	10**(-walker[4]) * vice.solar_z["fe"] / vice.solar_z["o"])
		# vice.yields.sneia.settings["fe"] = vice.yields.ccsne.settings["fe"] * (
		# 	10**(walker[4] - afe_eq) - 1)

		# output = self.sz.run(np.linspace(0, walker[3], N_TIMESTEPS + 1),
		# 	overwrite = True, capture = True)
		# diff = H3_UNIVERSE_AGE - walker[3]
		# model = []
		# for key in self.quantities:
		# 	if key == "lookback":
		# 		model.append(
		# 			# take into account offset between GSE stopping evolution
		# 			# and the present day
		# 			[m.log10(_ + diff) for _ in output.history[key][:-1]])
		# 	else:
		# 		model.append(output.history[key][1:])
		# model = np.array(model).T
		# weights = output.history["sfr"][1:]
		# norm = sum(weights)
		# weights = [_ / norm for _ in weights]
		# self.fd.model = model
		# self.fd.weights = weights
		# logp = self.fd()
		# print(logp)
		# return logp


if __name__ == "__main__":
	raw = np.genfromtxt(DATA_FILE)
	raw = np.array(list(filter(lambda x: x[4] >= 0.8 or np.isnan(x[4]), raw)))
	data = {
		"[fe/h]": np.array([row[0] for row in raw]),
		"[fe/h]_err": np.array([row[1] for row in raw]),
		# "[mg/fe]": np.array([row[2] for row in raw]),
		# "[mg/fe]_err": np.array([row[3] for row in raw]),
		"[o/fe]": np.array([row[2] for row in raw]),
		"[o/fe]_err": np.array([row[3] for row in raw]),
		"lookback": np.array([row[4] for row in raw]),
		"lookback_err": np.array([row[5] for row in raw])
	}
	log_prob = gsefit(data)
	pool = Pool(N_PROC)
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob, pool = pool)
	p0 = 10 * np.random.rand(N_WALKERS, N_DIM)
	# confine the [a/Fe] plateau to the allowed range to begin with
	for i in range(len(p0)):
		p0[i][4] /= 1000
		p0[i][5] /= 1000
	# 	while p0[i][4] < 0.1 or p0[i][4] > 0.8:
	# 		p0[i][4] = np.random.rand()
	start = time.time()
	state = sampler.run_mcmc(p0, N_BURNIN)
	sampler.reset()
	state = sampler.run_mcmc(state, N_ITERS)
	stop = time.time()
	print("MCMC time: ", stop - start)
	savechain(sampler, OUTFILE)

