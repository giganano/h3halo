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
]
RANGE = None
# RANGE = [
	# fiducial
	# (1.88, 2.14),
	# (8.6, 11.4),
	# (23.8, 26.2)
	# precise
	# (1.97, 2.05),
	# (9.5, 10.3),
	# (24.8, 25.2)
	# imprecise
	# (1.7, 2.4),
	# (5, 15),
	# (22, 28)
	# small
	# (1.5, 2.1),
	# (8, 16),
	# (23.4, 27.6)
	# noages
	# (1.6, 2.3),
	# (8.4, 12.6),
	# (23.4, 26)
	# large
	# (1.95, 2.05),
	# (9.6, 11.0),
	# (24.6, 25.5)
	# gse
# 	(0.92, 1.18),
# 	(15, 20),
# 	(17, 20)
	# some ages
# 	(1.92, 2.28),
# 	(8.6, 11.4),
# 	(24.1, 25.9),
	# lowered yields
# 	(1.85, 2.08),
# 	(2.6, 3.8),
# 	(7.5, 8.5)
# ]
TICKS = None
# TICKS = [
	# fiducial
	# [1.9, 2.0, 2.1],
	# [9, 10, 11],
	# [24, 25, 26]
	# precise
	# [1.98, 2, 2.02, 2.04],
	# [9.6, 9.8, 10, 10.2],
	# [24.9, 25, 25.1]
	# imprecise
	# [1.8, 2, 2.2],
	# [6, 8, 10, 12, 14],
	# [23, 25, 27]
	# small
	# [1.6, 1.8, 2.0],
	# [10, 12, 14],
	# [24, 25, 26, 27]
	# noages
	# [1.8, 2.0, 2.2],
	# [9, 10, 11, 12],
	# [24, 25, 26]
	# large
	# [1.96, 2, 2.04],
	# [10, 10.5],
	# [24.6, 25, 25.4]
	# gse
# 	[1., 1.1],
# 	[16, 17, 18, 19],
# 	[18, 19]
	# some ages
# 	(2, 2.1, 2.2),
# 	(9, 10, 11),
# 	(24.5, 25, 25.5)
	# lowered yields
# 	[1.9, 2.0],
# 	[3.0, 3.5],
# 	[7.5, 8.0, 8.5]
# ]
MAXLOGP_KWARGS = {
	"c": named_colors()["deepskyblue"],
	"marker": markers()["star"],
	"s": 150,
	"zorder": 100
}

raw = np.genfromtxt(FILENAME)
mcmc_chain = np.array([row[:-1] for row in raw])
logp = [row[-1] for row in raw]
idxmax = logp.index(max(logp))
DIM = len(mcmc_chain[0])

kwargs = {
	"labels": LABELS,
	# "quantiles": [0.30, 0.50, 0.70],
	"quantiles": [0.16, 0.50, 0.84],
	"show_titles": True,
	"color": named_colors()["black"],
	"truths": mcmc_chain[idxmax],
	# "truths": [2, 10, 25],
	"truth_color": named_colors()["crimson"]
}
if RANGE is not None: kwargs["range"] = RANGE
fig = corner.corner(mcmc_chain, **kwargs)
# fig.set_size_inches(15, 15)
# print(fig.size)
if TICKS is not None:
	for i in range(DIM):
		for j in range(DIM):
			if i >= j:
				fig.axes[DIM * i + j].xaxis.set_major_formatter(fsf("%g"))
				fig.axes[DIM * i + j].tick_params(labelrotation = 0)
				fig.axes[DIM * i + j].set_xticks(TICKS[j])
				if i != j:
					if not j: fig.axes[DIM * i + j].yaxis.set_major_formatter(
						fsf("%g"))
					fig.axes[DIM * i + j].set_yticks(TICKS[i])
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

