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
	r"$\tau_\star$ [Gyr]",
	r"$\eta$",
	r"$\tau_\text{tot}$ [Gyr]"
	# r"$100\times y_\text{O}^\text{CC}$",
	# r"$100\times y_\text{Fe}^\text{CC}$",
	# r"$100\times y_\text{Fe}^\text{Ia}$"
]
# RANGE = None
RANGE = [
	(1.5, 2.7),
	(18.5, 27.5),
	(14.4, 17.6),
	(6, 12)
# 	(1.7, 2.7),
# 	(5, 65),
# 	(10, 100),
# 	(8, 15),
# 	(0.5, 6),
# 	(0.05, 0.55),
# 	(0.0, 1)
]
# TICKS = None
TICKS = [
	[1.5, 2, 2.5],
	[20, 25],
	[15, 16, 17],
	[6, 8, 10, 12]
# 	[2.0, 2.5],
# 	[20, 40, 60],
# 	[20, 40, 60, 80],
# 	[10, 12, 14],
# 	[2, 4, 6],
# 	[0.2, 0.4],
# 	[0, 0.5, 1]
]
MAXLOGP_KWARGS = {
	"c": named_colors()["deepskyblue"],
	"marker": markers()["star"],
	"s": 150,
	"zorder": 100
}

raw = np.genfromtxt(FILENAME)
# raw = np.array(list(filter(lambda _: _[0] <= 5, raw)))
raw = np.array(list(filter(lambda _: not np.isinf(_[-1]), raw)))
mcmc_chain = np.array([row[:-1] for row in raw])
# for i in range(len(mcmc_chain)):
# 	for j in [-3, -2, -1]: mcmc_chain[i][j] *= 100
logp = [row[-1] for row in raw]
idxmax = logp.index(max(logp))
DIM = len(mcmc_chain[0])
# print(mcmc_chain[idxmax])

kwargs = {
	"labels": LABELS,
	# "quantiles": [0.30, 0.50, 0.70],
	"quantiles": [0.16, 0.50, 0.84],
	"show_titles": True,
	"color": named_colors()["black"],
	"truths": mcmc_chain[idxmax],
	# "truths": [2, 10, 25, 10, 1.5, 0.12, 0.17],
	"truth_color": named_colors()["crimson"]
}
# kwargs["truths"][-1] = 12.54
if RANGE is not None: kwargs["range"] = RANGE
fig = corner.corner(mcmc_chain, title_kwargs = {"fontsize": 15}, **kwargs)
# fig.set_size_inches(15, 15)
# print(fig.size)
if TICKS is not None:
	for i in range(DIM):
		for j in range(DIM):
			# if i >= j:
			# print(i, j, DIM, DIM * i + j)
			# print(fig.axes[DIM * i + j])
			if i == DIM - 1:
				fig.axes[DIM * i + j].xaxis.set_major_formatter(fsf("%g"))
				fig.axes[DIM * i + j].tick_params(labelrotation = 0)
			if j == 0 and i > 0:
				# print(i, j)
				# print(DIM * i)
				# print(fig.axes[DIM * i].__repr__())
				fig.axes[DIM * i].yaxis.set_major_formatter(fsf("%g"))
				fig.axes[DIM * i].tick_params(labelrotation = 0)
				# fig.axes[DIM * i].set_yticks(TICKS[i])
			if i >= j:
				fig.axes[DIM * i + j].set_xticks(TICKS[j])
				if i != j:
					fig.axes[DIM * i + j].set_yticks(TICKS[i])
					# if not j: fig.axes[DIM * i + j].yaxis.set_major_formatter(
					# 	fsf("%g"))
					# print(DIM * i + j)
					# print(mcmc_chain[idxmax][j])
					# print(mcmc_chain[idxmax][i])
					# fig.axes[DIM * i + j].scatter(
					# 	mcmc_chain[idxmax][j], mcmc_chain[idxmax][i],
					# 	**MAXLOGP_KWARGS)
else: pass
plt.tight_layout()
plt.subplots_adjust(hspace = 0, wspace = 0)
plt.savefig(sys.argv[2])

