
import numpy as np
import math as m
import vice
from vice.yields.presets import JW20

# def sfh(t):
# 	return m.exp(-t / 2)
def ifr(t):
	return m.exp(-t / 2)

with vice.singlezone(name = "mock", verbose = True) as sz:
	sz.func = ifr
	sz.mode = "ifr"
	sz.tau_star = 10
	sz.eta = 25
	sz.Mg0 = 0
	sz.run(np.linspace(0, 10, 1001), overwrite = True)

with vice.output("mock") as out:
	np.random.seed(0)
	totsfr = sum(out.history["sfr"])
	sfrfrac = [_ / totsfr for _ in out.history["sfr"]]
	indeces = np.random.choice(list(range(len(sfrfrac))), p = sfrfrac,
		size = 5000)
	with open("mock.dat", 'w') as data:
		data.write("# [fe/h]\t[o/fe]\tAge [Gyr]\n")
		for i in range(len(indeces)):
			feh = out.history["[fe/h]"][indeces[i]]
			ofe = out.history["[o/fe]"][indeces[i]]
			age = out.history["lookback"][indeces[i]]
			feh += np.random.normal(scale = 0.1)
			ofe += np.random.normal(scale = 0.1)
			age += np.random.normal(scale = 1)
			# age_diff = np.random.normal(scale = 1)
			# while age_diff < 0 and abs(age_diff) > age:
			# 	age_diff = np.random.normal(scale = 1)
			data.write("%.5e\t%.5e\t%.5e\n" % (feh, ofe, age))
		data.close()

