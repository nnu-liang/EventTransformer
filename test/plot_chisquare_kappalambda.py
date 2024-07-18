import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

kappa_lambda = [-8, -6, -4, -2, 0, 1, 2, 4, 6, 8, 10, 12]
S_values = [30, 25, 20, 15, 12, 10, 12, 15, 20, 25, 30, 35] 
S_sm = 10 
zeta_s = 0.001  

chi_squared = []
for S in S_values:
    chi2 = (S - S_sm)**2 / (S_sm + (zeta_s * S_sm)**2)
    chi_squared.append(chi2)

kappa_lambda_smooth = np.linspace(min(kappa_lambda), max(kappa_lambda), 300)
chi_squared_smooth = make_interp_spline(kappa_lambda, chi_squared)(kappa_lambda_smooth)

plt.figure(figsize=(10, 6))
plt.plot(kappa_lambda_smooth, chi_squared_smooth, color='lightblue', linestyle='-', linewidth=2, label='EvenTF')

plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
plt.axhline(y=3.84, color='black', linestyle='--', linewidth=1)

plt.text(12.5, 1, '68% CL', verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='black')
plt.text(12.5, 3.84, '95% CL', verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='black')

plt.legend()

plt.title(r'$\sqrt{s} =13 \mathrm{TeV}, 3000 \mathrm{fb}^{-1}, hh \rightarrow 4b $', fontsize=14)
plt.xlabel(r'$\kappa_{\lambda}$', fontsize=14)
plt.ylabel(r'$\chi^2$', fontsize=14)

plt.xticks(np.arange(-8, 14, 2), fontsize=12, fontweight='bold', color='black')
plt.yticks(np.arange(0, 6, 1), fontsize=12, fontweight='bold', color='black')
plt.xlim(-8, 12)
plt.ylim(0, 5)

plt.grid(True)

plt.savefig('chi_squared_plot.png')

plt.show()

