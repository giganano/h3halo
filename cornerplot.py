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
	r"$t_1$",
	r"$\sigma_1$",
	r"$A_2 / A_1$",
	r"$t_2$",
	r"$\sigma_2$",
	r"$\eta$",
	r"$\tau_{\star,0}$",
	r"$f_\text{burst}$",
	r"$\tau_\text{tot}$"
	# r"$\tau_\text{const}$ [Gyr]",
	# r"$\tau_\text{in}$ [Gyr]",
	# r"$\eta$",
	# r"$\tau_\star$ [Gyr]",
	# r"$\tau_\text{tot}$ [Gyr]",
	# r"$1000\times y_\text{Fe}^\text{CC}$",
	# r"$1000\times y_\text{Fe}^\text{Ia}$"
]
RANGE = None
# RANGE = [
	# (1, 3),
	# (0, 1.5),
	# (5, 12),
	# (8, 22),
	# (4, 7),
	# (0.6, 0.9),
	# (0.9, 1.7)
	###
	# (9, 15),
	# (10, 20),
	# (3, 8),
	# (0.7, 0.95),
	# (0.8, 1.3)
# ]
TICKS = None
# TICKS = [
# 	[1.6, 1.8, 2.0, 2.2],
# 	[200, 400],
# 	[200, 400, 600],
# 	[8, 9, 10, 11],
# 	[10, 20, 30, 40],
# 	[10, 20, 30, 40],
# 	[20, 40, 60]
# ]
MAXLOGP_KWARGS = {
	"c": named_colors()["deepskyblue"],
	"marker": markers()["star"],
	"s": 150,
	"zorder": 100
}

raw = np.genfromtxt(FILENAME)
raw = np.array(list(filter(lambda _: not np.isinf(_[-1]), raw)))
mcmc_chain = np.array([row[:-1] for row in raw])
# for i in range(len(mcmc_chain)):
# 	mcmc_chain[i][-1] *= 1000
# 	mcmc_chain[i][-2] *= 1000
# 	mcmc_chain[i][-3] *= 1000
logp = [row[-1] for row in raw]
idxmax = logp.index(max(logp))
DIM = len(mcmc_chain[0])

kwargs = {
	"labels": LABELS,
	"quantiles": [0.16, 0.50, 0.84],
	"show_titles": True,
	"color": named_colors()["black"],
	# "truths": mcmc_chain[idxmax],
	# "truths": [2, 10, 15, 10, 0.8, 1.1],
	# "truths": [2, 10, 15, 10, 0.8, 1.1, 10],
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

