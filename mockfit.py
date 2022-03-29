
from emcee import EnsembleSampler
import numpy as np
import math as m
import vice
from vice.yields.presets import JW20
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from utils import cov
from utils import exponential
import time

ENDTIME = 10
N_TIMESTEPS = 1000
N_WALKERS = 6
N_DIM = 3


class expifr_mcmc():

	def __init__(self, data, name = "mockfit"):
		self.quantities = list(data.keys())
		self.quantities = list(filter(lambda x: not x.endswith("_err"),
			self.quantities))
		# self.data = data
		self.data = np.array([data[key] for key in self.quantities]).T
		self.errors = np.array(
			[data["%s_err" % (key)] for key in self.quantities]).T
		# self.cov = np.array(cov(data))
		# self.invcov = np.linalg.inv(self.cov)
		self.elements = ["fe", "o"]
		self.nthreads = 2
		self.func = exponential()
		self.mode = "ifr"
		self.Mg0 = 0
		self.dt = ENDTIME / N_TIMESTEPS

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		with vice.singlezone(name = "mockfit", verbose = True) as sz:
			sz.elements = ["fe", "o"]
			sz.nthreads = 2
			sz.func = exponential(timescale = walker[0])
			sz.mode = "ifr"
			sz.Mg0 = 0
			sz.dt = ENDTIME / N_TIMESTEPS
			sz.tau_star = walker[1]
			sz.eta = walker[2]
			out = sz.run(np.linspace(0, ENDTIME, N_TIMESTEPS + 1),
				overwrite = True, capture = True)
		model = np.array([out.history[key][1:] for key in self.quantities]).T
		weights = out.history["sfr"][1:]

		# delta = np.zeros((len(self.data), len(model), len(self.quantities)))
		# delta = np.meshgrid(self.data, model)[0].T.reshape()
		# for i in range(len(self.data)):
		# 	for j in range(len(model)):
		# 		delta[i][j] = self.data[i] - model[j]
		# print("finished")
		delta = model[None, :] - self.data[:, None]
		# print(delta)
		# print(delta.shape)
		# invcovs = np.array(len(self.data) * [0])
		invcovs = np.zeros((len(self.data), len(self.quantities),
			len(self.quantities)))
		for i in range(len(self.data)):
			invcovs[i] = np.diag(self.errors[i]**2)
			invcovs[i] = np.linalg.inv(invcovs[i])
		# print(invcovs)
		arr = np.zeros((len(self.data), len(model)))
		logprob = 0
		for i in range(len(self.data)):
			arr[i] = -0.5 * np.diag(
				np.dot(invcovs[i].dot(delta[i].T).T, delta[i].T))
			logprob += logsumexp(arr[i], b = weights)
		return logprob

		# log_p_d_m = np.array(len(self.data) * [0.])
		# for i in range(len(self.data)):
		# 	cov = np.diag(self.errors[i]**2)
		# 	invcov = np.linalg.inv(cov)
		# 	arr = np.array(len(model) * [0.])
		# 	for j in range(len(arr)):
		# 		delta = self.data[i] - model[j]
		# 		arr[j] = -0.5 * np.matmul(np.matmul(
		# 			delta, invcov), delta.T)
		# 	log_p_d_m[i] = logsumexp(arr, b = weights)
		# 	# print(i, log_p_d_m[i])
		# 	sys.stdout.write("\r%d" % (i))
		# sys.stdout.write("\n")
		# log_p_d_m = np.sum(log_p_d_m)
		# print(log_p_d_m)
		# return log_p_d_m




if __name__ == "__main__":
	raw = np.genfromtxt("./mock.dat")
	data = {
		"[fe/h]": np.array([row[0] for row in raw]),
		"[fe/h]_err": np.array([row[1] for row in raw]),
		"[o/fe]": np.array([row[2] for row in raw]),
		"[o/fe]_err": np.array([row[3] for row in raw]),
		"lookback": np.array([row[4] for row in raw]),
		"lookback_err": np.array([row[5] for row in raw])
	}
	log_prob = expifr_mcmc(data)
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob)
	# start initial at known position anyway since this is a mock
	p0 = N_WALKERS * [None]
	for i in range(len(p0)):
		p0[i] = [2, 10, 25]
		# p0[i] = [25, 10, 2]
		# p0[i]  = [3, 15, 25]
		for j in range(len(p0[i])):
			p0[i][j] += np.random.normal(scale = 0.1 * p0[i][j])
	p0 = np.array(p0)
	# print(p0)
	start = time.time()
	# state = sampler.run_mcmc(p0, 1, skip_initial_state_check = True)
	state = sampler.run_mcmc(p0, 200, skip_initial_state_check = True)
	stop = time.time()
	print("MCMC time: ", stop - start)
	samples = sampler.get_chain()
	samples = np.concatenate(tuple([samples[i] for i in range(len(samples))]))
	# print(samples)
	np.savetxt("mockchain.out", samples, fmt = "%.5e")

