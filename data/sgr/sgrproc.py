
import numpy as np

if __name__ == "__main__":
	raw = np.genfromtxt("sgrchem_orig.dat")
	out = np.array([row[:4] for row in raw])
	np.savetxt("sgrchem.dat", out, fmt = "%.4f", delimiter = '\t',
		header = "[Fe/H] [Fe/H]_err [alpha/Fe] [alpha/Fe]_err")

