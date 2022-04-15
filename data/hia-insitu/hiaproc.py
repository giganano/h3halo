
import numpy as np
raw = np.genfromtxt("hiachem.dat")

with open("hiachem.dat", 'w') as out:
	out.write("# [Fe/H]\t[Fe/H]_err\t[Mg/Fe]\t[Mg/Fe]_err\t")
	out.write("Log(age)\tLog(age)_err\n")
	for i in range(len(raw)):
		if raw[i][-2] != -999:
			raw[i][-2] -= 9 # log(yr) -> log(Gyr)
			for j in range(len(raw[i])): out.write("%.4f\t" % (raw[i][j]))
			out.write("\n")
		else:
			for j in range(len(raw[i]) - 2): out.write("%.4f\t" % (raw[i][j]))
			out.write("nan\tnan\n")
