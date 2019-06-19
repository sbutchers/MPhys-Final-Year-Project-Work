import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

### r vs. n_s plot using certainty data from 2015 PLANCK data release. ###
PlanckTT_lowP_1sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP_1sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
PlanckTT_lowP_2sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP_2sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
PlanckTT_lowP_BKP_1sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP+BKP_1sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
PlanckTT_lowP_BKP_2sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP+BKP_2sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
lensing_ext_1sig = np.genfromtxt('ns_r_Data/+lensing+ext_1sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
lensing_ext_2sig = np.genfromtxt('ns_r_Data/+lensing+ext_2sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.plot(PlanckTT_lowP_2sig[:, 0], PlanckTT_lowP_2sig[:, 1], 'r-', label='PlanckTT+lowP')
ax.plot(PlanckTT_lowP_BKP_2sig[:, 0], PlanckTT_lowP_BKP_2sig[:, 1], 'g-', label='PlanckTT+lowP+BKP')
ax.plot(lensing_ext_2sig[:, 0], lensing_ext_2sig[:, 1], 'b-', label='+lensing+ext')
ax.fill_between(PlanckTT_lowP_1sig[:, 0], 0, PlanckTT_lowP_1sig[:, 1], alpha=0.5, facecolor='red')
ax.fill_between(PlanckTT_lowP_2sig[:, 0], 0, PlanckTT_lowP_2sig[:, 1], alpha=0.5, facecolor='red')
ax.fill_between(PlanckTT_lowP_BKP_1sig[:, 0], 0, PlanckTT_lowP_BKP_1sig[:, 1], alpha=0.5, facecolor='green')
ax.fill_between(PlanckTT_lowP_BKP_2sig[:, 0], 0, PlanckTT_lowP_BKP_2sig[:, 1], alpha=0.5, facecolor='green')
ax.fill_between(lensing_ext_1sig[:, 0], 0, lensing_ext_1sig[:, 1], alpha=0.5, facecolor='blue')
ax.fill_between(lensing_ext_2sig[:, 0], 0, lensing_ext_2sig[:, 1], alpha=0.5, facecolor='blue')
ax.set_xlim([0.945, 1.0])
ax.set_ylim([0.0, 0.26])
ax.set_xlabel(r'$n_{s}$')
ax.set_ylabel(r'$r$')
ax.legend(prop={'size':10}, loc="upper right")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("ns_r_interp.pdf", dpi=2000, bbox_inches='tight')
