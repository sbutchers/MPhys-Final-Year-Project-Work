### Import modules needed with start time for timing the program. ###
import time
start_time = time.time()
import numpy as np
from scipy.integrate import odeint as od
from scipy import interpolate
import matplotlib.pyplot as plt
from textwrap import wrap
from scipy.optimize import curve_fit

#############################################################################################################################

### Define the two 1st order ODEs from the second order ODE derived on paper.
def derivs(phi, N): # return derivatives of the array phi
    a = 0.5
    b = -3.0
    c = -6.0
    return np.array([ phi[1], a*phi[1]**3 + b*phi[1] + (phi[1]**2)/phi[0] + c/phi[0] ])

### Provide initial conditions and N-space interval to solve ODE
Nend = 75.0
Nfolds = np.linspace(0.0, Nend, 50000)
phiStart = 17.251
PhiInit = np.array([phiStart, -1e-5])
phis = od(derivs, PhiInit, Nfolds)

### Plot of the solution in phi from the ODE
plt.figure()
plt.plot(Nfolds, phis[:,0], 'b-', linewidth=0.25) # phi solution is 1st column of phis
plt.title(r'A plot of $\phi[N]$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
plt.xlabel('No. of e-folds, N', fontsize=12)
plt.ylabel(r'Scalar field, $\phi[N]$ (dimensionless)', fontsize=12)
plt.xlim([0.0, Nend])
plt.ylim([-1.0, phiStart+1])
plt.grid('on')
plt.savefig("PhiNPlt.pdf", dpi=2000, bbox_inches=0)
plt.close()

#############################################################################################################################

### Solution for n_ic
## Definitions of potentials
m = 1e-6
V = 0.5 * m**2 * (phis[:, 0])**2
Vprime = m**2 * phis[:, 0]
## Definition of Hubble parameter H and Hubble radius squared = (aH)^2
H = np.sqrt(V / (3 - 0.5 * (phis[:, 1]**2)))
a = np.exp(Nfolds)
HubRadSqu = (a * H)**2
## Definition of wavenumbers k and n_ic value.
def find_nearest(array,value): # function to find index in array closest to desired value
    idx = (np.abs(array-value)).argmin()
    return idx

n_exit = Nend - np.arange(60.0, 49.5, -0.5)
print n_exit
k = np.empty(len(n_exit))
phi_exit = np.empty(len(n_exit))
for i in range(0, len(n_exit)):
    H_n = H[find_nearest(Nfolds, n_exit[i])]
    a_n = np.exp(n_exit[i])
    k[i] = a_n * H_n
    phi_exit[i] = phis[find_nearest(Nfolds, n_exit[i]), 0]

print "Phi at horizon exit is:", phi_exit
C = -2.0 + np.log(2) + np.euler_gamma

n_s_slowroll = 1.0 - ((8.0) / (phi_exit**2)) - ((56.0 + 96.0*C)/(3.0 * phi_exit**4))
print "Slow-roll prediction of n_s is:", n_s_slowroll

kmax = np.amax(k)
Cq = 1e4
HubRadNeeded = kmax / Cq
Index = find_nearest(np.sqrt(HubRadSqu), HubRadNeeded)
n_ic = Nfolds[Index]

#############################################################################################################################

### Solve the scalar and tensor equations of motion for the perturbations from each wave-mode
### k, introducing new initial conditions for each value of k. (Mukhanov-Sasaki equation recast
### in e-fold time. Also need to reduce values for this new N range starting at n_ic.)

### SET UP COEFFICIENTS NEEDED FOR EACH K VALUE IN THE EOMS. ###
index = find_nearest(Nfolds, n_ic)
nFolds = Nfolds[index:]

## Definition of terms used in the Mukhanov-Sasaki equation
INVaHsquared = 1.0/HubRadSqu[index:]
aH_ic = np.sqrt(INVaHsquared[0])
NprimeprimeBYaH2 = 1.0 - ((phis[index:, 1])**2 / 2)
phis2prime = (
    ((phis[index:, 1])**3 / 2.0) -
    (3.0 * phis[index:, 1]) -
    (6.0 / phis[index:, 0]) +
    ((phis[index:, 1])**2 / phis[index:, 0])
)
phis3prime = (
    ((3.0 * phis2prime * (phis[index:, 1])**2) / 2.0) -
    (3.0 * phis2prime) +
    ((6.0 * phis[index:, 1]) / (phis[index:, 0])**2) +
    ((2.0 * phis2prime * phis[index:, 1]) / (phis[index:, 0])) -
    ((phis[index:, 1])**3 / (phis[index:, 0])**2)
)
z2primeBYz = (
    1 +
    ((2 * phis2prime) / (phis[index:, 1])) +
    (phis3prime / phis[index:, 1])
)
z1primeBYz = (
    1 +
    (phis2prime / phis[index:, 1])
)

# Define arrays of desired coefficient values.
v1coeff = ((phis[index:, 1])**2 / 2) - 1
v0coeff1 = -1.0 * INVaHsquared
v0coeff2 = (z2primeBYz + (NprimeprimeBYaH2 * z1primeBYz))

# Produce interpolation functions of each coefficent.
s1coeffI = interpolate.InterpolatedUnivariateSpline(nFolds, v1coeff)
s0coeffI1 = interpolate.InterpolatedUnivariateSpline(nFolds, v0coeff1)
s0coeffI2 = interpolate.InterpolatedUnivariateSpline(nFolds, v0coeff2)

## Definition of scalar perturbation EOM in e-fold time as a function.
def PerturbDeriv(v_k, N, k):
    return np.array([ v_k[1], s1coeffI(N)*v_k[1] + s0coeffI1(N)*(k**2)*v_k[0] + s0coeffI2(N)*v_k[0] ])
## Definition of tensor perturbation EOM in e-fold time.
def TensPerturbDeriv(v_K, N, k):
    return np.array([ v_K[1], s1coeffI(N)*v_K[1] + s0coeffI1(N)*(k**2)*v_K[0] - s1coeffI(N)*v_K[0] + v_K[0] ])

# Define empty lists for tensor to scalar ratio and scalar spectrum at the e-fold no. where N = 45.
rRatio40 = []
LNsSpec40 = []

## Iterate through k values with their corresponding n_ic chosen above solving the equation above each time.
for i in range(0, len(k)):
    ### CALCULATION OF SCALAR PERTURBATIONS. ###
    # Set-up initial conditions in v_k
    v_k_icIMAG = (HubRadNeeded) / np.sqrt(2 * (k[i])**3)
    v_k_icREAL = (1.0) / (np.sqrt(2 * k[i]))
    v_kPrime_icIMAG = (1.0 / (np.sqrt(2 * k[i]))) * ((HubRadNeeded/k[i]) - (k[i]/HubRadNeeded))
    v_kPrime_icREAL = 1.0/(np.sqrt(2 * k[i]))
    v_initialsIMAG = np.array([v_k_icIMAG, v_kPrime_icIMAG])
    v_initialsREAL = np.array([v_k_icREAL, v_kPrime_icREAL])

    # Imaginary solution of v_k using ODEINT on the ODE
    v_ksIMAG = od( PerturbDeriv, v_initialsIMAG, nFolds, args=(k[i],) )

    # Real solution of v_k using ODEINT on the ODE
    v_ksREAL = od( PerturbDeriv, v_initialsREAL, nFolds, args=(k[i],) )

    # Combine imaginary and real solutions of v_k to find the Mukhanov variable.
    v_kSolution = v_ksREAL**2 + v_ksIMAG**2
    ABSv_kSolution = np.sqrt(v_kSolution)

    # Plot of Mukhanov variable, |v_k| vs. no of e-folds, N.
    plt.figure()
    plt.plot(nFolds, ABSv_kSolution[:,0], 'b-', linewidth=0.25)
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the absolute value of the Mukhanov variable $|v_{k}|$ vs. no of e-folds N for k = %d.' % k[i], 80)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Mukhanov variable, $|v_{k}|$')
    plt.savefig("ScalarPerturbations/AbsMukhanovVar/absV_kNPlt_%d.pdf" % i, dpi=2000, bbox_inches=0)
    plt.close()

    # Plot of |v_k|^2 vs. N
    ModSqV_k = (ABSv_kSolution[:,0])**2

    plt.figure()
    plt.plot(nFolds, ModSqV_k, 'b-', linewidth=0.25)
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the absolute value of the Mukhanov variable squared $|v_{k}|^{2}$ vs. no of e-folds N for k = %d.' % k[i], 80)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Mukhanov variable squared, $|v_{k}|^{2}$')
    plt.savefig("ScalarPerturbations/AbsMukhanovVarSqua/absV_k^2NPlt_%d.pdf" % i, dpi=2000, bbox_inches=0)
    plt.close()

    # Dimensionless scalar power spectrum calculation.
    globals()['ScalPowSpec' + str(i)] = (k[i]**3 / (2 * np.pi**2)) * (ModSqV_k / (np.exp(nFolds) * phis[Index:, 1])**2)
    BunchDavieSol = (1.0 / (4.0 * (np.pi* phis[index:, 1])**2)) * ((k[i] / np.exp(nFolds))**2 + (H[index:] / nFolds)**2)

    plt.figure()
    plt.plot(nFolds, globals()['ScalPowSpec' + str(i)], 'b-', linewidth=0.25, label=r'$\Delta_{\mathcal{R}}^{2}(k) = \frac{k^{3}}{2\pi^{2}} \left(\frac{|v_{k}|}{a \phi^{\prime}}\right)^{2}$')
    plt.plot(nFolds, BunchDavieSol, 'g-', label = 'Bunch-Davies Solution')
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the scalar power spectrum, $\Delta_{\mathcal{R}}^{2}(k)$ vs. no of e-folds, N for k = %d.' % k[i], 80)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Scalar power spectrum, $\Delta_{\mathcal{R}}^{2}(k)$')
    plt.legend(loc="best")
    plt.savefig("ScalarPerturbations/ScalarPowerSpectrum/ScalPowSpec_%d.pdf" % i, dpi=2000, bbox_inches=0)
    plt.close()


    ### CALCULATION OF TENSOR PERTURBATIONS. ###
    # Set-up initial conditions in v_k
    v_K_icIMAG = HubRadNeeded/np.sqrt(2 * (k[i])**3)
    v_K_icREAL = 1.0/(np.sqrt(2 * k[i]))
    v_KPrime_icIMAG = (1.0 / (np.sqrt(2 * k[i]))) * ((HubRadNeeded/k[i]) - (k[i]/HubRadNeeded))
    v_KPrime_icREAL = 1.0/(np.sqrt(2 * k[i]))
    v_K_initialsIMAG = np.array([v_k_icIMAG, v_kPrime_icIMAG])
    v_K_initialsREAL = np.array([v_k_icREAL, v_kPrime_icREAL])

    # Imaginary solution of v_k using ODEINT on the ODE
    v_KsIMAG = 1j * od( TensPerturbDeriv, v_K_initialsIMAG, nFolds, args=(k[i],) )

    # Real solution of v_k using ODEINT on the ODE
    v_KsREAL = od( TensPerturbDeriv, v_K_initialsREAL, nFolds, args=(k[i],) )

    # Combine imaginary and real solutions of v_k to find the Mukhanov variable.
    v_KSolution = v_KsREAL + v_KsIMAG
    ABSv_KSolution = abs(v_KSolution)

    # Plot of Mukhanov variable, |v_k| vs. no of e-folds, N.
    plt.figure()
    plt.plot(nFolds, ABSv_KSolution[:,0], 'b-', linewidth=0.25)
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the absolute value of the Mukhanov variable $|v_{k}|$ vs. no of e-folds N for k = %d.' % k[i], 80)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Mukhanov variable, $|v_{k}|$')
    plt.savefig("TensorPerturbations/AbsMukhanovVar/absV_kNPlt_%d.pdf" % i, dpi=2000, bbox_inches=0)
    plt.close()

    # Plot of |v_k|^2 vs. N
    ModSqV_K = (ABSv_KSolution[:,0])**2

    plt.figure()
    plt.plot(nFolds, ModSqV_K, 'b-', linewidth=0.25)
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the absolute value of the Mukhanov variable squared $|v_{k}|^{2}$ vs. no of e-folds N for k = %d.' % k[i], 80)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Mukhanov variable squared, $|v_{k}|^{2}$')
    plt.savefig("TensorPerturbations/AbsMukhanovVarSqua/absV_k^2NPlt_%d.pdf" % i, dpi=2000, bbox_inches=0)
    plt.close()

    # Dimensionless tensor power spectrum calculation.
    globals()['TensPowSpec' + str(i)] = (k[i]**3 / (2 * np.pi**2)) * ((8 * ModSqV_K) / (np.exp(nFolds))**2)

    plt.figure()
    plt.plot(nFolds, globals()['TensPowSpec' + str(i)], 'b-', linewidth=0.25, label=r'$\Delta_{t}^{2}(k) = \frac{k^{3}}{2\pi^{2}} \frac{8 |v_{k}|^{2}}{a^{2}}$')
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the tensor power spectrum, $\Delta_{t}^{2}(k)$ vs. no of e-folds, N for k = %d.' % k[i], 80)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Tensor power spectrum, $\Delta_{t}^{2}(k)$')
    plt.legend(loc="best")
    plt.savefig("TensorPerturbations/TensorPowerSpectrum/TensPowSpec_%d.pdf" % i, dpi=2000, bbox_inches=0)
    plt.close()

    # Tensor to scalar ratio r.
    globals()['r' + str(i)] = globals()['TensPowSpec' + str(i)] / globals()['ScalPowSpec' + str(i)]
    plt.figure()
    plt.plot(nFolds, globals()['r' + str(i)], 'b-', linewidth=0.25, label=r'$r = \frac{\Delta_{t}^{2}(k)}{\Delta_{s}^{2}(k)}$')
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the tensor to scalar ratio for k = %d.' % k[i], 80)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel('Tensor to scalar ratio r')
    plt.legend(loc="best")
    plt.savefig("TensorScalarRatio/TensScalRatio_%d.pdf" % i, dpi=2000, bbox_inches=0)
    plt.close()

    # Extract tensor to scalar ratio and scalar power spectrum when N = 45
    index40 = find_nearest(nFolds, 45.0)
    rRatio40.append(globals()['r' + str(i)][index40])
    print "Tensor to scalar ratio for k = %d is: %.3f" % (k[i], globals()['r' + str(i)][index40])
    LNsSpec40.append(globals()['ScalPowSpec' + str(i)][index40])
    print "Scalar power spectrum for k = %d is: %g" % (k[i], globals()['ScalPowSpec' + str(i)][index40])

#############################################################################################################################

### SPECTRAL INDEX ###
LNsSpec40 = np.log(LNsSpec40)
print LNsSpec40
lnK = np.log(k)
print lnK

## Define quadratic function for fitting and fit to data using curve_fit. ##
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

## User-found parameters for the fit. ##
p_0 = [1, 1, 1]

## Fitting procedure. ##
popt, pcov = curve_fit(quadratic, lnK[4:], LNsSpec40[4:], p_0)
print popt

# Gradient: d(Ln(ScalSpec))/d(Ln(k)):
stuff = ((2.0 * popt[0] * lnK) + popt[1]) + 1.0
print rRatio40
print stuff

## Plot the fitted quadratic curve and the ln(ScalSpec) vs. ln(k) curve. ##
plt.figure()
plt.plot(lnK, LNsSpec40)
plt.plot(lnK, quadratic(lnK, popt[0], popt[1], popt[2]))
plt.xlabel(r'$\left| \ln(k) \right|$')
plt.ylabel(r'$\left| \ln\left(\Delta_{\mathcal{R}}^{2}(k)\right) \right|$')
plt.savefig("ScalSpecKplt.pdf", dpi=2000, bbox_inches=0)
plt.close()

#############################################################################################################################

### r vs. n_s plot using certainty data from 2015 PLANCK data release. ###
PlanckTT_lowP_1sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP_1sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
PlanckTT_lowP_2sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP_2sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
PlanckTT_lowP_BKP_1sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP+BKP_1sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
PlanckTT_lowP_BKP_2sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP+BKP_2sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
lensing_ext_1sig = np.genfromtxt('ns_r_Data/+lensing+ext_1sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
lensing_ext_2sig = np.genfromtxt('ns_r_Data/+lensing+ext_2sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.plot(PlanckTT_lowP_2sig[:, 0], PlanckTT_lowP_2sig[:, 1], 'r-', label='PlanckTT+lowP')
ax.plot(PlanckTT_lowP_BKP_2sig[:, 0], PlanckTT_lowP_BKP_2sig[:, 1], 'g-', label='PlanckTT+lowP+BKP')
ax.plot(lensing_ext_2sig[:, 0], lensing_ext_2sig[:, 1], 'b-', label='+lensing+ext')
ax.plot(stuff, rRatio40, 'kx-', label=r'$\phi^{2}$ Data - Quadratic Fit')
ax.plot(n_s_slowroll, rRatio40, 'mx-', label=r'$\phi^{2}$ Data - 2nd order slow roll')
ax.fill_between(PlanckTT_lowP_1sig[:, 0], 0, PlanckTT_lowP_1sig[:, 1], alpha=0.5, facecolor='red')
ax.fill_between(PlanckTT_lowP_2sig[:, 0], 0, PlanckTT_lowP_2sig[:, 1], alpha=0.5, facecolor='red')
ax.fill_between(PlanckTT_lowP_BKP_1sig[:, 0], 0, PlanckTT_lowP_BKP_1sig[:, 1], alpha=0.5, facecolor='green')
ax.fill_between(PlanckTT_lowP_BKP_2sig[:, 0], 0, PlanckTT_lowP_BKP_2sig[:, 1], alpha=0.5, facecolor='green')
ax.fill_between(lensing_ext_1sig[:, 0], 0, lensing_ext_1sig[:, 1], alpha=0.5, facecolor='blue')
ax.fill_between(lensing_ext_2sig[:, 0], 0, lensing_ext_2sig[:, 1], alpha=0.5, facecolor='blue')
ax.set_xlim([0.945, 1.0])
ax.set_ylim([0.0, 0.26])
ax.set_title('\n'.join(wrap(r'A plot of the tensor to scalar ratio, $r_{0.002}$ against the spectral index, $n_{s}$.', 50)))
ax.set_xlabel(r'$n_{s}$')
ax.set_ylabel(r'$r_{0.002}$')
ax.legend(prop={'size':10}, loc="upper right")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("ns_r.pdf", dpi=2000, bbox_inches=0)

############################################################################################################################

### Print out time taken to run program. ###
print("--- %s seconds ---" % (time.time() - start_time))
