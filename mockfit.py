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
from src import fit_driver
import numpy as np
import math as m
import vice
from vice.yields.presets import JW20
vice.yields.ccsne.settings['o'] /= 3
vice.yields.ccsne.settings['fe'] /= 3
vice.yields.sneia.settings['fe'] /= 3
from utils import cov
from utils import exponential
import time
import sys
import os

ENDTIME = 10
N_TIMESTEPS = int(sys.argv[6])
N_WALKERS = int(sys.argv[4])
N_DIM = 3


def invcov(errors):
	nancheck = [m.isnan(_) for _ in errors]
	if any(nancheck):
		smallerrors = list(filter(lambda _: not m.isnan(_), errors))
		smallcov = np.diag([_**2 for _ in smallerrors])
		smallinvcov = np.linalg.inv(smallcov)
		invcov = np.zeros((len(errors), len(errors)))
		nrow = 0
		for i in range(len(invcov)):
			ncol = 0
			if nancheck[i]:
				for j in range(len(invcov[i])): invcov[i][j] = float("nan")
			else:
				for j in range(len(invcov[i])):
					if nancheck[j]:
						invcov[i][j] = float("nan")
					else:
						invcov[i][j] = smallinvcov[nrow][ncol]
						ncol += 1
				nrow += 1
		return invcov
	else:
		cov = np.diag([_**2 for _ in errors])
		return np.linalg.inv(cov)


class expifr_mcmc:

	def __init__(self, data, name = sys.argv[3]):
		self._sz = vice.singlezone(name = name)
		self._sz.elements = ["fe", "o"]
		self._sz.func = exponential()
		self._sz.mode = "ifr"
		self._sz.Mg0 = 0
		self._sz.nthreads = 2
		self._sz.dt = ENDTIME / N_TIMESTEPS
		# super().__init__(name = name, **kwargs)
		# self.elements = ["fe", "o"]
		# self.func = exponential()
		# self.mode = "ifr"
		# self.Mg0 = 0
		# self.nthreads = 2
		# self.dt = ENDTIME / N_TIMESTEPS

		self.quantities = list(data.keys())
		self.quantities = list(filter(lambda x: not x.endswith("_err"),
			self.quantities))
		sample = np.array([data[key] for key in self.quantities]).T
		errors = np.array(
			[data["%s_err" % (key)] for key in self.quantities]).T
		invcovs = len(sample) * [None]
		for i in range(len(sample)): invcovs[i] = invcov(errors[i])

		self.fd = fit_driver(sample, invcovs)

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		self._sz.name = "%s%s" % (sys.argv[3], os.getpid())
		self._sz.timescale = walker[0]
		self._sz.tau_star = walker[1]
		self._sz.eta = walker[2]
		out = self._sz.run(np.linspace(0, ENDTIME, N_TIMESTEPS + 1),
			overwrite = True, capture = True)
		print("walker: [%.5e, %.5e, %.5e] " % (walker[0], walker[1], walker[2]))
		# self.func.timescale = walker[0]
		# self.tau_star = walker[1]
		# self.eta = walker[2]
		# out = super().run(np.linspace(0, ENDTIME, N_TIMESTEPS + 1),
		# 	overwrite = True, capture = True)
		model = []
		for key in self.quantities:
			if key == "lookback":
				model.append([m.log10(_) for _ in out.history[key][:-1]])
			else:
				model.append(out.history[key][1:])
		model = np.array(model).T
		# model = np.array([out.history[key][1:] for key in self.quantities]).T
		weights = out.history["sfr"][1:]
		norm = sum(weights)
		weights = [_ / norm for _ in weights]

		self.fd.model = model
		self.fd.weights = weights

		result = self.fd()
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
		# "lookback": np.array([m.log10(row[4]) for row in raw]),
		# "lookback_err": np.array([m.log10(row[5]) for row in raw])
	}
	log_prob = expifr_mcmc(data)
	pool = Pool(int(sys.argv[8]))
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob, pool = pool)
	# print(dir(sampler))
	# print(sampler.pool)
	# quit()
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
	samples = sampler.get_chain()
	logprob = sampler.get_log_prob()
	samples = np.concatenate(tuple([samples[i] for i in range(len(samples))]))
	logprob = np.concatenate(tuple([logprob[i] for i in range(len(logprob))]))
	logprob = [[logprob[_]] for _ in range(len(logprob))]
	out = np.append(samples, logprob, axis = 1)
	af = sum(sampler.acceptance_fraction) / N_WALKERS
	np.savetxt(sys.argv[2], out, fmt = "%.5e",
		header = "acceptance fraction: %.5e" % (af))

