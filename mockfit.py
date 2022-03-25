
from emcee import EnsembleSampler
import numpy as np
import math as m
import vice
from vice.yields.presets import JW20
from utils import cov
from utils import exponential

N_TIMESTEPS = 500
N_WALKERS = 10
N_DIM = 3


class expifr_mcmc(vice.singlezone):

	def __init__(self, data, name = "mockfit", **kwargs):
		super().__init__(name = name, **kwargs)
		self.data = data
		self.cov = np.array(cov(data))
		self.invcov = np.linalg.inv(self.cov)
		self.mode = "ifr"
		self.func = exponential()
		self.Mg0 = 0

	def __call__(self, walker):
		if any([_ < 0 for _ in walker]): return -float("inf")
		self.func.timescale = walker[0]
		self.tau_star = walker[1]
		self.eta = walker[2]
		out = super().run(np.linspace(0, 10, N_TIMESTEPS + 1),
			overwrite = True, capture = True)
		model = np.array(len(out.history["time"]) * [None])
		keys = list(self.data.keys())
		# print(keys)
		for i in range(len(model)):
			model[i] = len(keys) * [0.]
			for j in range(len(keys)):
				model[i][j] = out.history[keys[j]][i]
		model = model[1:]
		# print(model)
		chisq = 0
		for i in range(len(self.data[keys[0]])):
			p = np.array([self.data[_][i] for _ in keys])
			# print("============")
			# print(p)
			lb_diff = [abs(p[-1] - model[_][-1]) for _ in range(len(model))]
			# print(lb_diff)
			idx = lb_diff.index(min(lb_diff))
			# print(idx)
			predicted = np.array(model[idx])
			# print(predicted)
			dp = np.array([p[_] - predicted[_] for _ in range(len(keys))])
			# print(dp)
			chisq += np.dot(np.dot(dp, self.invcov), dp.T)
		# print("==")
		# print(walker)
		# print(chisq)
		return -0.5 * chisq



if __name__ == "__main__":
	raw = np.genfromtxt("./mock.dat")
	data = {
		"[fe/h]": [row[0] for row in raw],
		"[o/fe]": [row[1] for row in raw],
		"lookback": [row[2] for row in raw]
	}
	log_prob = expifr_mcmc(data, verbose = True)
	sampler = EnsembleSampler(N_WALKERS, N_DIM, log_prob)
	# start initial at known position anyway since this is a mock
	p0 = N_WALKERS * [None]
	for i in range(len(p0)):
		# p0[i] = [2, 10, 25]
		p0[i] = [25, 10, 2]
		for j in range(len(p0[i])):
			p0[i][j] += np.random.normal(scale = 0.1)
	p0 = np.array(p0)
	# print(p0)
	state = sampler.run_mcmc(p0, 2, skip_initial_state_check = True)
	samples = sampler.get_chain()
	print(samples)









