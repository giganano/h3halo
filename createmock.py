
import vice
import numpy as np
import math as m
from src.mock import create_mock
vice.yields.ccsne.settings['o'] = 0.01
vice.yields.sneia.settings['o'] = 0
vice.yields.ccsne.settings['fe'] = 0.0008
vice.yields.sneia.settings['fe'] = 0.0011


def ifr(t):
	return m.exp(-t / 2)


with vice.singlezone(name = "./mocksamples/n100", verbose = True) as sz:
	sz.elements = ["fe", "o"]
	sz.nthreads = 2
	sz.func = ifr
	sz.mode = "ifr"
	sz.Mg0 = 0
	sz.tau_star = 15
	sz.eta = 10
	sz.dt = 0.01
	create_mock(sz, "./mocksamples/n100.dat", n_stars = 100, n_ages = 20)

