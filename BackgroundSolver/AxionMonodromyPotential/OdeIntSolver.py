import numpy as np
from scipy.integrate import odeint as od
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

### Define the two 1st order ODEs from the second order ODE derived on paper.
def derivs(phi, N): # return derivatives of the array phi
    a = 0.5
    b = -3.0
    c = 0.5
    d = -3.0
    return np.array([ phi[1], a*phi[1]**3 + b*phi[1] + (c * phi[1]**2)/phi[0] + d/phi[0] ])

### Provide initial conditions and N-space interval to solve ODE
Nfolds = np.linspace(0.0, 51.5, 10000)
PhiInit = np.array([10.0, 0.0])
phis = od(derivs, PhiInit, Nfolds)

### Plot of the solution in phi from the ODE
plt.figure()
plt.plot(Nfolds, phis[:,0], 'b-', linewidth=0.25) # phi solution is 1st column of phis
#plt.title(r'A plot of $\phi[N]$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
plt.xlabel('No. of e-folds, N', fontsize=12)
plt.ylabel(r'Scalar field, $\phi[N]$ (dimensionless)', fontsize=12)
plt.xlim([0.0, 51.5])
plt.ylim([-2.0, 10.0])
plt.grid('on')
plt.savefig("PhiNPlt.pdf", dpi=2000, bbox_inches='tight')
plt.close()

### Plot of ln(rho) against the no. of e-folds
m = 1e-4
V = m**3 * phis[:,0]
Rho = V / (1 - (phis[:,1]**2)/6 )
lnRho = np.log(Rho)

plt.figure()
plt.plot(Nfolds, lnRho, 'b-', linewidth=0.25) # Plotting ln(rho) from phi' solution.
#plt.title(r'A plot of ln($\rho$) vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
plt.xlabel('No. of e-folds, N', fontsize=10)
plt.ylabel(r'ln($\rho$)', fontsize=10)
plt.xlim([0.0, 65.0])
plt.grid('on')
plt.savefig("RhoNPlt.pdf", dpi=2000, bbox_inches='tight')
plt.close()

### Plot of the inflation condition against no. of e-folds
H = (V / (3 - 0.5*(phis[:,1]**2)))**0.5
InflationCondition = 1.0 - (0.5* phis[:,1]**2) # Inflation condition 18.3 from PDP book (=adotdot/aH^2 here).

plt.figure()
plt.plot(Nfolds, InflationCondition, 'b-', linewidth=0.25) # Inflation ends when Inflation condition = 1.
#plt.title(r'A plot of $\frac{1}{H^{2}} \left(\frac{\ddot{a}}{a}\right)$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
plt.xlabel('No. of e-folds, N', fontsize=10)
plt.ylabel(r'Inflation condition, $\frac{1}{H^{2}} \left(\frac{\ddot{a}}{a}\right)$', fontsize=10)
plt.xlim([0.0, 59.0])
plt.grid('on')
plt.savefig("InflCondNPlt.pdf", dpi=2000, bbox_inches='tight')
plt.close()

### A plot of the Hubble parameter against the no. of e-folds.
plt.figure()
plt.plot(Nfolds, H, 'b-', linewidth=0.25)
plt.xlabel('No of e-folds, N', fontsize=10)
plt.ylabel('Hubble parameter, H', fontsize=10)
plt.grid('on')
plt.savefig("HubbleParam.pdf", dpi=2000, bbox_inches='tight')
plt.close()

# ### Plot of the curvature slow-roll parameter eta against the no. of e-folds.
# eta = 2.0 / (phis[:,0])**2

# plt.figure()
# plt.plot(Nfolds, eta, 'b-', linewidth=0.25)
# #plt.title(r'A plot of $\eta$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
# plt.xlabel('No. of e-folds, N', fontsize=10)
# plt.ylabel(r'Curvature parameter, $\eta$', fontsize=10)
# plt.xlim([0.0, 60.0])
# plt.ylim([0.0, 10.0])
# plt.grid('on')
# plt.savefig("EtaNPlt.pdf", dpi=2000, bbox_inches='tight')
# plt.close()

### Plot of the slope slow-roll paramter epsilon against the no. of e-folds.
epsilon = 0.5 / (phis[:,0])**2

plt.figure()
plt.plot(Nfolds, epsilon, 'b-', linewidth=0.25)
#plt.title(r'A plot of $\epsilon$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
plt.xlabel('No. of e-folds, N', fontsize=10)
plt.ylabel(r'Slope parameter, $\epsilon$', fontsize=10)
plt.xlim([0.0, 65.0])
plt.ylim([0.0, 10.0])
plt.grid('on')
plt.savefig("EpsilonNPlt.pdf", dpi=2000, bbox_inches='tight')
plt.close()

### Plot of a combination of phi, inflation condition and ...
func = lambda phi : 0.5 / (phi**2) - 1
phi_initial_guess = 57.2
phi_solution = abs(fsolve(func, phi_initial_guess))
def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx
index = find_nearest(epsilon, 1.0)
index2 = find_nearest(phis[:,1], -np.sqrt(2.0))
print epsilon[index]
print InflationCondition[index2]

INDEX = find_nearest(phis[:,0], 1e250)
print np.amax(phis[:,0])
print phis[INDEX,0]
print phis[INDEX,1]
print Nfolds[INDEX]
print H[INDEX]
print InflationCondition[INDEX]

plt.figure()
plt.plot(Nfolds, phis[:,0], 'b-', linewidth=0.5, label=r"$\phi[N]$ solution")
plt.plot(Nfolds, InflationCondition, 'r-', linewidth=0.5, label=r"$\frac{1}{H^{2}} \left(\frac{\ddot{a}}{a}\right)$ solution")
plt.plot(Nfolds, epsilon, 'm--', linewidth=2.0, label=r'$\epsilon$ solution')
# plt.plot(Nfolds, eta, 'g-', linewidth=0.5, label=r'$\eta$ solution')
plt.axvline(Nfolds[index], 0, 1, color='k', linestyle='--', lw=1.0)
plt.annotate('Slow-roll end at N = %.3f' % (Nfolds[index]), xy=(Nfolds[index], epsilon[index]), xytext=(50.3, -0.7), bbox=dict(boxstyle="square", fc="w"), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Inflation condition end at N = %.3f' % (Nfolds[index2]), xy=(Nfolds[index2], InflationCondition[index2]), xytext=(50.7, -2.5), bbox=dict(boxstyle="square", fc="w"), arrowprops=dict(facecolor='black', shrink=0.05))
#plt.title(r'A plot of $\phi[N]$, $\epsilon$ and $\eta$ vs. N for $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
plt.xlabel('No. of e-folds, N', fontsize=12)
plt.ylabel(r'$\phi[N]$, $\epsilon$ and $\eta$ (all dimensionless)', fontsize=10)
plt.xlim([50.0, 51.5])
plt.ylim([-4.0, 4.0])
plt.legend(prop={'size':14}, loc="lower left")
plt.grid('on')
plt.savefig('CombinedNPlt.pdf', dpi=2000, bbox_inches='tight')
plt.close()

### Plot of 1/aH against the no. of e-folds

plt.figure()
plt.plot(Nfolds[1:], np.log(H[0]/(np.exp(Nfolds[1:]) * H[1:])), 'b-', linewidth=0.25)
plt.xlabel('No. of e-folds, N', fontsize=10)
plt.ylabel(r'$\ln\left(\frac{a_{0}H_{0}}{aH}\right)$', fontsize=10)
plt.axvline(Nfolds[index2], 0, 1, color='k', linestyle='--', lw=1.0)
plt.xlim([0.0, 52.0])
plt.ylim([-60.0, 0.0])
plt.grid('on')
plt.savefig("HubbleRadiusInfCondition.pdf", dpi=2000, bbox_inches='tight')
plt.close()
