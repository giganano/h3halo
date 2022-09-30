r"""
ARGV
----
1) The name of the file containing the MCMC data. Log likehood values assumed
	to be in the far right column.
2) The name of the output image.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter as fsf
from plots.mpltoolkit import load_mpl_presets, named_colors, markers
import corner
import numpy as np
import sys
load_mpl_presets()

FILENAME = sys.argv[1]
LABELS = [
	r"$\tau_\text{in}$ [Gyr]",
	r"$\eta$",
	r"$\tau_\star$ [Gyr]",
	r"$\tau_\text{tot}$ [Gyr]",
	r"$10^4\times y_\text{Fe}^\text{CC}$",
	r"$10^3\times y_\text{Fe}^\text{Ia}$",
	r"$10^2\times y_\alpha^\text{CC}$"
]
RANGE = None
# RANGE = [
# 	(1.3, 2.4),
# 	(8.5, 11.5),
# 	(10, 18),
# 	(6.5, 12.5),
# 	(7.2, 9.4),
# 	(0.85, 1.25)
###
	# (0.4, 1.6),
	# (4, 13),
	# (11, 22),
	# (4, 7),
	# (6.2, 9.6),
	# (0.8, 1.7)
###
	# (0, 12),
	# (20, 70),
	# (10, 90),
	# (1, 6)
###
# 	(1.4, 2.4),
# 	(10, 120),
# 	(30, 180),
# 	(7, 13),
# 	(10, 90),
# 	(1, 12),
# 	(1, 10)
# ]
TICKS = None
# TICKS = [
# 	[1.5, 2.0],
# 	[9, 10, 11],
# 	[12, 14, 16],
# 	[8, 10, 12],
# 	[7.5, 8, 8.5, 9],
# 	[0.9, 1, 1.1, 1.2]
###
#	[0.5, 1, 1.5],
# 	[6, 8, 10, 12],
# 	[15, 20],
# 	[5, 6],
# 	[7, 8, 9],
# 	[1.0, 1.5]
###
# 	[1.5, 2.0],
# 	[50, 100],
# 	[50, 100, 150],
# 	[8, 10, 12],
# 	[20, 40, 60, 80],
# 	[5, 10],
# 	[2, 4, 6, 8]
# ]
MAXLOGP_KWARGS = {
	"c": named_colors()["deepskyblue"],
	"marker": markers()["star"],
	"s": 150,
	"zorder": 100
}

raw = np.genfromtxt(FILENAME)
raw = np.array(list(filter(lambda _: not np.isinf(_[-1]), raw)))
# raw = np.array(list(filter(lambda _: _[0] <= 10, raw)))
# raw = np.array(list(filter(lambda _: _[-2] <= 0.002, raw)))
# raw = np.array(list(filter(lambda _: _[-3] <= 1e-3, raw)))
# raw = np.array(list(filter(lambda _: _[-2] <= 6, raw)))
# raw = np.array(list(filter(lambda _: RANGE[0][0] <= _[0] <= RANGE[0][1], raw)))
# raw = np.array(list(filter(lambda _: _[-3] < 12e-4, raw)))
# raw = np.array(list(filter(lambda _: RANGE[1][0] <= _[1] <= RANGE[1][1], raw)))
print(len(raw))
mcmc_chain = np.array([row[:-1] for row in raw])
for i in range(len(mcmc_chain)):
	mcmc_chain[i][-1] *= 1000
	mcmc_chain[i][-2] *= 10000
	# mcmc_chain[i][-3] *= 10000
logp = [row[-1] for row in raw]
idxmax = logp.index(max(logp))
DIM = len(mcmc_chain[0])
print(mcmc_chain[idxmax])

kwargs = {
	"labels": LABELS,
	"quantiles": [0.16, 0.50, 0.84],
	"show_titles": True,
	"color": named_colors()["black"],
	"truths": mcmc_chain[idxmax],
	# "truths": [2, 10, 15, 10, 8, 1.1],
	# "truths": [2, 10, 15, 10, 0.8, 1.1, 1],
	"truth_color": named_colors()["crimson"]
}
if RANGE is not None: kwargs["range"] = RANGE
fig = corner.corner(mcmc_chain, title_kwargs = {"fontsize": 15}, **kwargs)
if TICKS is not None:
	for i in range(DIM):
		for j in range(DIM):
			if i == DIM - 1:
				fig.axes[DIM * i + j].xaxis.set_major_formatter(fsf("%g"))
				fig.axes[DIM * i + j].tick_params(labelrotation = 0)
			if j == 0 and i > 0:
				fig.axes[DIM * i].yaxis.set_major_formatter(fsf("%g"))
				fig.axes[DIM * i].tick_params(labelrotation = 0)
			if i >= j:
				fig.axes[DIM * i + j].set_xticks(TICKS[j])
				if i != j:
					fig.axes[DIM * i + j].set_yticks(TICKS[i])
					# fig.axes[DIM * i + j].scatter(
					# 	mcmc_chain[idxmax][j], mcmc_chain[idxmax][i],
					# 	**MAXLOGP_KWARGS)
else: pass
plt.tight_layout()
plt.subplots_adjust(hspace = 0, wspace = 0)
plt.savefig(sys.argv[2])

