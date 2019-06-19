### Import modules needed with start time for timing the program. ###
import time
start_time = time.time()
import numpy as np
from scipy.integrate import odeint as od
from scipy import interpolate
import matplotlib.pyplot as plt
from textwrap import wrap
#from scipy.optimize import curve_fit

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
## Definition of wavenumbers k
def find_nearest(array,value): # function to find index in array closest to desired value
    idx = (np.abs(array-value)).argmin()
    return idx
# Use k/aH = 1 at N = 50, 60 to define k values.
H50 = H[find_nearest(Nfolds, 50.0)]
H60 = H[find_nearest(Nfolds, 60.0)]
a50 = np.exp(10.0)
a60 = 1.0
kLow = H50*a50
kHigh = H60*a60
#k = np.linspace(kLow, kHigh, 10)
#kmax = np.amax(k)
## Solve for n_ic given HubRad, k and arbitrary constant C_q
Cq = 1e5
#HubRadNeeded = kmax / Cq
#index = find_nearest(HubRadSqu, HubRadNeeded)
#n_ic = Nfolds[index]

## New attempt at assigning n_ic and k values.
n_ic = Nend - np.array([65.0, 64.0, 63.0, 62.0, 61.0, 60.0,
                         59.0, 58.0, 57.0, 56.0, 55.0])
k = np.empty(len(n_ic))
for i in range(0, len(n_ic)):
    H_n = H[find_nearest(Nfolds, n_ic[i])]
    a_n = np.exp(n_ic[i])
    k[i] = a_n * H_n


#############################################################################################################################

### Solve the scalar and tensor equations of motion for the perturbations from each wave-mode
### k, introducing new initial conditions for each value of k. (Mukhanov-Sasaki equation recast
### in e-fold time. Also need to reduce values for this new N range starting at n_ic.)

### SCALAR EQUATION & PERTURBATIONS.
## Definition of terms used in the Mukhanov-Sasaki equation
#INVaHsquared = 1.0/HubRadSqu[index:]
##nFolds = Nfolds[index:]
#NprimeprimeBYaH2 = 1.0 - ((phis[index:, 1])**2 / 2)
#phis2prime = (
#    ((phis[index:, 1])**3 / 2) -
#    (3 * phis[index:, 1]) -
#    (6 / phis[index:, 0]) +
#    ((phis[index:, 1])**2 / phis[index:, 0])
#)
#phis3prime = (
#    ((3 * phis2prime * (phis[index:, 1])**2)/2) -
#    (3 * phis2prime) +
#    ((6 * phis[index:, 1]) / (phis[index:, 0])**2) +
#    ((2 * phis2prime * phis[index:, 1]) / (phis[index:, 0])) -
#    ((phis[index:, 1])**3 / (phis[index:, 0]))
#)
#z2primeBYz = (
#    1 +
#    ((2 * phis2prime) / (phis[index:, 1])) +
#    (phis3prime / phis[index:, 1])
#)
#z1primeBYz = (
#    1 +
#    (phis2prime / phis[index:, 1])
#)
#
### Make plots of above quantities to check for values are consistent with expectations.
#plt.figure()
#plt.plot(Nfolds, H)
#plt.title(r'A plot of $\frac{1}{H^{2}}$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
#plt.xlabel("No. of e-folds, N")
#plt.ylabel(r'Inverse Hubble parameter squared, $\frac{1}{H^{2}}$ [s$^{-1}$].')
#plt.savefig("ConsistencyChecks/HubbleConst.pdf", dpi=2000, bbox_inches=0)
#plt.close()
#
#plt.figure()
#plt.plot(nFolds, phis[index:, 1])
#plt.title(r'A plot of $\phi^{\prime}$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
#plt.xlabel("No. of e-folds, N")
#plt.ylabel(r'1st derivative of phi, $\frac{d\phi}{dN}$.')
#plt.savefig("ConsistencyChecks/PhiPrime.pdf", dpi=2000, bbox_inches=0)
#plt.close()
#
#plt.figure()
#plt.plot(nFolds, phis2prime)
#plt.title(r'A plot of $\phi^{\prime \prime}$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
#plt.xlabel("No. of e-folds, N")
#plt.ylabel(r'2nd derivative of phi, $\frac{d^{2}\phi}{dN^{2}}$.')
#plt.savefig("ConsistencyChecks/Phi2Prime.pdf", dpi=2000, bbox_inches=0)
#plt.close()
#
#plt.figure()
#plt.plot(nFolds, phis3prime)
#plt.title(r'A plot of $\phi^{\prime \prime \prime}$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
#plt.xlabel("No. of e-folds, N")
#plt.ylabel(r'3rd derivative of phi, $\frac{d^{3}\phi}{dN^{3}}$.')
#plt.savefig("ConsistencyChecks/Phi3Prime.pdf", dpi=2000, bbox_inches=0)
#plt.close()
#
#plt.figure()
#plt.plot(nFolds, 1/np.sqrt(INVaHsquared))
#plt.title(r'A plot of $\frac{1}{(aH)^{2}}$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.', y=1.02)
#plt.xlabel("No. of e-folds, N")
#plt.ylabel(r'Inverse Hubble radius squared, $\frac{1}{(aH)^{2}}$.')
#plt.savefig("ConsistencyChecks/aHsquared.pdf", dpi=2000, bbox_inches=0)
#plt.close()
#
#plt.figure()
#plt.plot(
#    nFolds,
#    z2primeBYz,
#    label=r'$\frac{z^{\prime \prime}}{z} = 1 + \frac{2\phi^{\prime \prime}}{\phi^{\prime}} + \frac{\phi^{\prime \prime \prime}}{\phi^{\prime}}$'
#)
#plt.title(
#    r'A plot of $\frac{z^{\prime \prime}}{z}$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.',
#    y=1.02
#)
#plt.xlabel("No. of e-folds, N")
#plt.ylabel(r'$\frac{z^{\prime \prime}}{z}$')
#plt.legend(prop={'size':14}, loc="upper left")
#plt.savefig("ConsistencyChecks/z2primebyz.pdf", dpi=2000, bbox_inches=0)
#plt.close()
#
#plt.figure()
#plt.plot(
#    nFolds,
#    z1primeBYz,
#    label=r'$\frac{z^{\prime}}{z} = 1 + \frac{\phi^{\prime \prime}}{\phi^{\prime}}$'
#)
#plt.title(
#    r'A plot of $\frac{z^{\prime}}{z}$ vs. N from $V = \frac{1}{2}m^{2}\phi^{2}$.',
#    y=1.02
#)
#plt.xlabel("No. of e-folds, N")
#plt.ylabel(r'$\frac{z^{\prime}}{z}$')
#plt.legend(prop={'size':14}, loc="upper left")
#plt.savefig("ConsistencyChecks/z1primebyz.pdf", dpi=2000, bbox_inches=0)
#plt.close()
#
### Coefficients of the ODE now need to be interpolated in order for the odeint package to
### work correctly. Here I also produce three plots to compare the interpolated result with
### the original array of exact values.
#
## Define arrays of desired coefficient values.
#v1coeff = ((phis[index:, 1])**2 / 2) - 1
#v0coeff1 = -1.0 * INVaHsquared
#v0coeff2 = (z2primeBYz + (NprimeprimeBYaH2 * z1primeBYz))
#
## Produce interpolation functions of each coefficent.
#s1coeffI = interpolate.InterpolatedUnivariateSpline(nFolds, v1coeff)
#s0coeffI1 = interpolate.InterpolatedUnivariateSpline(nFolds, v0coeff1)
#s0coeffI2 = interpolate.InterpolatedUnivariateSpline(nFolds, v0coeff2)
#
## Check each interpolation is good with plots of each coefficent.
#plt.figure()
#plt.plot(nFolds, s1coeffI(nFolds), label='Interpolation values of coefficient.', linewidth=2.0)
#plt.plot(nFolds, v1coeff, 'r-', label='Actual values of coefficient.', linewidth=0.75)
#plt.title(r'Interpolation of $v_{k}^{\prime}$ coefficient in the ODE.')
#plt.xlabel("No. of e-folds, N")
#plt.ylabel(r'$v_{k}^{\prime}$ coefficient.')
#plt.legend(prop={'size':14}, loc="upper left")
#plt.savefig("ConsistencyChecks/v1coeff.pdf", dpi=2000, bbox_inches=0)
#plt.close()
#
#plt.figure()
#plt.plot(nFolds, s0coeffI1(nFolds), label='Interpolation values of coefficient.', linewidth=2.0)
#plt.plot(nFolds, v0coeff1, 'r-', label='Actual values of coefficient.', linewidth=0.75)
#plt.title(r'Interpolation of the first $v_{k}$ coefficient in the ODE.')
#plt.xlabel("No. of e-folds, N")
#plt.ylabel(r'$v_{k}$ coefficient 1.')
#plt.legend(prop={'size':14}, loc="lower right")
#plt.savefig("ConsistencyChecks/v0coeff1.pdf", dpi=2000, bbox_inches=0)
#plt.close()
#
#plt.figure()
#plt.plot(nFolds, s0coeffI2(nFolds), label='Interpolation values of coefficient.', linewidth=2.0)
#plt.plot(nFolds, v0coeff2, 'r-', label='Actual values of coefficient.', linewidth=0.75)
#plt.title(r'Interpolation of the second $v_{k}$ coefficient in the ODE.')
#plt.xlabel("No. of e-folds, N")
#plt.ylabel(r'$v_{k}$ coefficient 2.')
#plt.legend(prop={'size':14}, loc="upper left")
#plt.savefig("ConsistencyChecks/v0coeff2.pdf", dpi=2000, bbox_inches=0)
#plt.close()

#############################################################################################################################

## Definition of scalar perturbation EOM in e-fold time as a function.
def PerturbDeriv(v_k, N, k):
    return np.array([ v_k[1], s1coeffI(N)*v_k[1] + s0coeffI1(N)*(k**2)*v_k[0] + s0coeffI2(N)*v_k[0] ])
## Definition of tensor perturbation EOM in e-fold time.
def TensPerturbDeriv(v_K, N, k):
    return np.array([ v_K[1], s1coeffI(N)*v_K[1] + s0coeffI1(N)*(k**2)*v_K[0] - s1coeffI(N)*v_K[0] + v_K[0] ])

## Iterate through k values chosen above solving the equation above each time.
a_end = np.exp(Nend)
kappa = 8 * np.pi
#aH_ic = np.sqrt(INVaHsquared[0])

#ScalPowSpec = np.empty([len(nFolds), len(k)])

for i in range(0, len(k)):
    index = find_nearest(Nfolds, n_ic[i])
    nFolds = Nfolds[index:]

    ## Definition of terms used in the Mukhanov-Sasaki equation
    INVaHsquared = 1.0/HubRadSqu[index:]
    aH_ic = np.sqrt(INVaHsquared[0])
    #nFolds = Nfolds[index:]
    NprimeprimeBYaH2 = 1.0 - ((phis[index:, 1])**2 / 2)
    phis2prime = (
        ((phis[index:, 1])**3 / 2) -
        (3 * phis[index:, 1]) -
        (6 / phis[index:, 0]) +
        ((phis[index:, 1])**2 / phis[index:, 0])
    )
    phis3prime = (
        ((3 * phis2prime * (phis[index:, 1])**2)/2) -
        (3 * phis2prime) +
        ((6 * phis[index:, 1]) / (phis[index:, 0])**2) +
        ((2 * phis2prime * phis[index:, 1]) / (phis[index:, 0])) -
        ((phis[index:, 1])**3 / (phis[index:, 0]))
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

    # Set-up initial conditions in v_k
    v_k_icIMAG = 0.0
    v_k_icREAL = (kappa / (np.sqrt(k[i]) * a_end)) * np.exp(Nend - n_ic[i])
    v_kPrime_icIMAG = ((-kappa / (np.sqrt(k[i]) * a_end)) * np.exp(Nend - n_ic[i])) * (k[i] * aH_ic)
    v_kPrime_icREAL = ((-kappa / (np.sqrt(k[i]) * a_end)) * np.exp(Nend - n_ic[i]))
    v_initialsIMAG = np.array([v_k_icIMAG, v_kPrime_icIMAG])
    v_initialsREAL = np.array([v_k_icREAL, v_kPrime_icREAL])

    # Imaginary solution of v_k using ODEINT on the ODE
    v_ksIMAG = 1j * od( PerturbDeriv, v_initialsIMAG, nFolds, args=(k[i],) )

    # Real solution of v_k using ODEINT on the ODE
    v_ksREAL = od( PerturbDeriv, v_initialsREAL, nFolds, args=(k[i],) )

    # Combine imaginary and real solutions of v_k to find the Mukhanov variable.
    v_kSolution = v_ksREAL + v_ksIMAG
    ABSv_kSolution = abs(v_kSolution)

    # Plot of Mukhanov variable, |v_k| vs. no of e-folds, N.
    plt.figure()
    plt.plot(nFolds, ABSv_kSolution[:,0], 'b-', linewidth=0.25)
    #plt.xlim([0.0, 63.0])
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the absolute value of the Mukhanov variable $|v_{k}|$ vs. no of e-folds N.', 60)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Mukhanov variable, $|v_{k}|$')
    plt.savefig("ScalarPerturbations/AbsMukhanovVar/absV_kNPlt_k_(%f).pdf" % k[i], dpi=2000, bbox_inches=0)
    plt.close()

    # Plot of |v_k|^2 vs. N
    ModSqV_k = (ABSv_kSolution[:,0])**2

    plt.figure()
    plt.plot(nFolds, ModSqV_k, 'b-', linewidth=0.25)
    #plt.xlim([0.0, 63.0])
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the absolute value of the Mukhanov variable squared $|v_{k}|^{2}$ vs. no of e-folds N.', 60)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Mukhanov variable squared, $|v_{k}|^{2}$')
    plt.savefig("ScalarPerturbations/AbsMukhanovVarSqua/absV_k^2NPlt_k_(%f).pdf" % k[i], dpi=2000, bbox_inches=0)
    plt.close()

    # Dimensionless scalar power spectrum calculation.
    ScalPowSpec = (k[i]**3 / (2 * np.pi**2)) * (ModSqV_k / (np.exp(nFolds) * phis[index:, 1])**2)

    plt.figure()
    plt.plot(nFolds, ScalPowSpec, 'b-', linewidth=0.25, label=r'$\Delta_{\mathcal{R}}^{2}(k) = \frac{k^{3}}{2\pi^{2}} \left(\frac{|v_{k}|}{a \phi^{\prime}}\right)^{2}$')
    #plt.xlim([0.0, 63.0])
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the scalar power spectrum, $\Delta_{\mathcal{R}}^{2}(k)$ vs. no of e-folds, N.', 80)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Scalar power spectrum, $\Delta_{\mathcal{R}}^{2}(k)$')
    plt.legend(loc="best")
    plt.savefig("ScalarPerturbations/ScalarPowerSpectrum/ScalPowSpec_k_(%f).pdf" % k[i], dpi=2000, bbox_inches=0)
    plt.close()

    print ScalPowSpec
    print ScalPowSpec.shape

    # Set-up initial conditions in v_k
    v_K_icIMAG = 0.0
    v_K_icREAL = (kappa / (np.sqrt(k[i]) * a_end)) * np.exp(Nend - n_ic[i])
    v_K_Prime_icIMAG = ((-kappa / (np.sqrt(k[i]) * a_end)) * np.exp(Nend - n_ic[i])) * (k[i] * aH_ic)
    v_K_Prime_icREAL = ((-kappa / (np.sqrt(k[i]) * a_end)) * np.exp(Nend - n_ic[i]))
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
    #plt.xlim([0.0, 63.0])
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the absolute value of the Mukhanov variable $|v_{k}|$ vs. no of e-folds N.', 60)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Mukhanov variable, $|v_{k}|$')
    plt.savefig("TensorPerturbations/AbsMukhanovVar/absV_kNPlt_k_(%f).pdf" % k[i], dpi=2000, bbox_inches=0)
    plt.close()

    # Plot of |v_k|^2 vs. N
    ModSqV_K = (ABSv_KSolution[:,0])**2

    plt.figure()
    plt.plot(nFolds, ModSqV_K, 'b-', linewidth=0.25)
    #plt.xlim([0.0, 63.0])
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the absolute value of the Mukhanov variable squared $|v_{k}|^{2}$ vs. no of e-folds N.', 60)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Mukhanov variable squared, $|v_{k}|^{2}$')
    plt.savefig("TensorPerturbations/AbsMukhanovVarSqua/absV_k^2NPlt_k_(%f).pdf" % k[i], dpi=2000, bbox_inches=0)
    plt.close()

    # Dimensionless tensor power spectrum calculation.
    TensPowSpec = (k[i]**3 / (2 * np.pi**2)) * ((8 * ModSqV_K) / (np.exp(nFolds))**2)

    plt.figure()
    plt.plot(nFolds, TensPowSpec, 'b-', linewidth=0.25, label=r'$\Delta_{t}^{2}(k) = \frac{k^{3}}{2\pi^{2}} \frac{8 |v_{k}|^{2}}{a^{2}}$')
    #plt.xlim([0.0, 63.0])
    plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the tensor power spectrum, $\Delta_{t}^{2}(k)$ vs. no of e-folds, N.', 80)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel(r'Tensor power spectrum, $\Delta_{t}^{2}(k)$')
    plt.legend(loc="best")
    plt.savefig("TensorPerturbations/TensorPowerSpectrum/TensPowSpec_k_(%f).pdf" % k[i], dpi=2000, bbox_inches=0)
    plt.close()

    print TensPowSpec
    print TensPowSpec.shape

    r = TensPowSpec / ScalPowSpec
    plt.figure()
    plt.plot(nFolds, r, 'b-', linewidth=0.25, label=r'$r = \frac{\Delta_{t}^{2}(k)}{\Delta_{s}^{2}(k)}$')
    #plt.xlim([0.0, 63.0])
    #plt.yscale('log')
    plt.grid('on')
    plt.title('\n'.join(wrap(r'A plot of the tensor to scalar ratio for k = %f.' % k[i], 80)))
    plt.xlabel("No. of e-folds, N")
    plt.ylabel('Tensor to scalar ratio r')
    plt.legend(loc="best")
    plt.savefig("TensorScalarRatio/TensScalRatio_k_(%f).pdf" % k[i], dpi=2000, bbox_inches=0)
    plt.close()



#############################################################################################################################

### TENSOR EQUATION AND PERTURBATIONS.

## Note: The equation of motion for the tensor perturbations uses a lot of the same
## quantities derived and found before for the scalar case so there is no need to
## do any more interpolations for the coefficients in the equation of motion.

## Definition of tensor perturbation EOM in e-fold time.
#def TensPerturbDeriv(v_K, N, k):
#    return np.array([ v_K[1], s1coeffI(N)*v_K[1] + s0coeffI1(N)*(k**2)*v_K[0] - s1coeffI(N)*v_K[0] + v_K[0] ])
#
#TensPowSpec = np.empty([len(nFolds), len(k)])
#
### Iterate through k values chosen above solving the equation above each time.
#for i in range(0, len(k)):
#    INDEX = find_nearest(Nfolds, n_ic[i])
#    nFolds = Nfolds[INDEX:]
#    # Set-up initial conditions in v_k
#    v_K_icIMAG = 0.0
#    v_K_icREAL = (kappa / (np.sqrt(k[i]) * a_end)) * np.exp(Nend - n_ic[i])
#    v_K_Prime_icIMAG = ((-kappa / (np.sqrt(k[i]) * a_end)) * np.exp(Nend - n_ic[i])) * (k[i] * aH_ic)
#    v_K_Prime_icREAL = ((-kappa / (np.sqrt(k[i]) * a_end)) * np.exp(Nend - n_ic[i]))
#    v_K_initialsIMAG = np.array([v_k_icIMAG, v_kPrime_icIMAG])
#    v_K_initialsREAL = np.array([v_k_icREAL, v_kPrime_icREAL])
#
#    # Imaginary solution of v_k using ODEINT on the ODE
#    v_KsIMAG = 1j * od( TensPerturbDeriv, v_K_initialsIMAG, nFolds, args=(k[i],) )
#
#    # Real solution of v_k using ODEINT on the ODE
#    v_KsREAL = od( TensPerturbDeriv, v_K_initialsREAL, nFolds, args=(k[i],) )
#
#    # Combine imaginary and real solutions of v_k to find the Mukhanov variable.
#    v_KSolution = v_KsREAL + v_KsIMAG
#    ABSv_KSolution = abs(v_KSolution)
#
#    # Plot of Mukhanov variable, |v_k| vs. no of e-folds, N.
#    plt.figure()
#    plt.plot(nFolds, ABSv_KSolution[:,0], 'b-', linewidth=0.25)
#    #plt.xlim([0.0, 63.0])
#    plt.yscale('log')
#    plt.grid('on')
#    plt.title('\n'.join(wrap(r'A plot of the absolute value of the Mukhanov variable $|v_{k}|$ vs. no of e-folds N.', 60)))
#    plt.xlabel("No. of e-folds, N")
#    plt.ylabel(r'Mukhanov variable, $|v_{k}|$')
#    plt.savefig("TensorPerturbations/AbsMukhanovVar/absV_kNPlt_k_(%f).pdf" % k[i], dpi=2000, bbox_inches=0)
#    plt.close()
#
#    # Plot of |v_k|^2 vs. N
#    ModSqV_K = (ABSv_KSolution[:,0])**2
#
#    plt.figure()
#    plt.plot(nFolds, ModSqV_K, 'b-', linewidth=0.25)
#    #plt.xlim([0.0, 63.0])
#    plt.yscale('log')
#    plt.grid('on')
#    plt.title('\n'.join(wrap(r'A plot of the absolute value of the Mukhanov variable squared $|v_{k}|^{2}$ vs. no of e-folds N.', 60)))
#    plt.xlabel("No. of e-folds, N")
#    plt.ylabel(r'Mukhanov variable squared, $|v_{k}|^{2}$')
#    plt.savefig("TensorPerturbations/AbsMukhanovVarSqua/absV_k^2NPlt_k_(%f).pdf" % k[i], dpi=2000, bbox_inches=0)
#    plt.close()
#
#    # Dimensionless tensor power spectrum calculation.
#    TensPowSpec[:, i] = (k[i]**3 / (2 * np.pi**2)) * ((8 * ModSqV_K) / (np.exp(nFolds))**2)
#
#    plt.figure()
#    plt.plot(nFolds, TensPowSpec[:, i], 'b-', linewidth=0.25, label=r'$\Delta_{t}^{2}(k) = \frac{k^{3}}{2\pi^{2}} \frac{8 |v_{k}|^{2}}{a^{2}}$')
#    #plt.xlim([0.0, 63.0])
#    plt.yscale('log')
#    plt.grid('on')
#    plt.title('\n'.join(wrap(r'A plot of the tensor power spectrum, $\Delta_{t}^{2}(k)$ vs. no of e-folds, N.', 80)))
#    plt.xlabel("No. of e-folds, N")
#    plt.ylabel(r'Tensor power spectrum, $\Delta_{t}^{2}(k)$')
#    plt.legend(loc="best")
#    plt.savefig("TensorPerturbations/TensorPowerSpectrum/TensPowSpec_k_(%f).pdf" % k[i], dpi=2000, bbox_inches=0)
#    plt.close()
#
#print TensPowSpec
#print TensPowSpec.shape

#############################################################################################################################

#### TENSOR TO SCALAR RATIO ###
#r = TensPowSpec / ScalPowSpec
#for i in range(0, len(k)):
#    plt.figure()
#    plt.plot(nFolds, r[:, i], 'b-', linewidth=0.25, label=r'$r = \frac{\Delta_{t}^{2}(k)}{\Delta_{s}^{2}(k)}$')
#    #plt.xlim([0.0, 63.0])
#    plt.yscale('log')
#    plt.grid('on')
#    plt.title('\n'.join(wrap(r'A plot of the tensor to scalar ratio for k = %f.' % k[i], 80)))
#    plt.xlabel("No. of e-folds, N")
#    plt.ylabel('Tensor to scalar ratio r')
#    plt.legend(loc="best")
#    plt.savefig("TensorScalarRatio/TensScalRatio_k_(%f).pdf" % k[i], dpi=2000, bbox_inches=0)
#    plt.close()
#
#### SPECTRAL INDEX ###
### Find the values of the spectral power spectrum at N = 30 and ln(k) values. ##
#index30 = find_nearest(nFolds, 30.0)
#LNsSpec30 = np.log(ScalPowSpec[index30, :])
#lnK = np.log(k)
#
### Define quadratic curve for fitting and fit to data using curve_fit. ##
#def quadratic(x, a, b, c):
#    return a * x**2 + b * x + c
#
### User-found parameters for the fit. ##
#p_0 = [1, 1, 30]
#
### Fitting procedure. ##
#popt, pcov = curve_fit(quadratic, abs(lnK), abs(LNsSpec30), p_0)
#print popt
#
### Plot the fitted quadratic curve and the ln(ScalSpec) vs. ln(k) curve. ##
#plt.figure()
#plt.plot(abs(lnK), abs(LNsSpec30))
#plt.plot(abs(lnK), quadratic(abs(lnK), popt[0], popt[1], popt[2]))
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel(r'$\left| \ln(k) \right|$')
#plt.ylabel(r'$\left| \ln\left(\Delta_{\mathcal{R}}^{2}(k)\right) \right|$')
#plt.savefig("ScalSpecKplt.pdf", dpi=2000, bbox_inches=0)
#plt.close()
#
##############################################################################################################################
#
#### r vs. n_s plot using certainty data from 2015 PLANCK data release. ###
#PlanckTT_lowP_1sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP_1sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
#PlanckTT_lowP_2sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP_2sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
#PlanckTT_lowP_BKP_1sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP+BKP_1sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
#PlanckTT_lowP_BKP_2sig = np.genfromtxt('ns_r_Data/PlanckTT+lowP+BKP_2sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
#lensing_ext_1sig = np.genfromtxt('ns_r_Data/+lensing+ext_1sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
#lensing_ext_2sig = np.genfromtxt('ns_r_Data/+lensing+ext_2sig.txt', delimiter='\t', skip_header=6, usecols=(0,1))
#
#fig = plt.figure(figsize=(6,6))
#ax = fig.add_subplot(111)
#ax.plot(PlanckTT_lowP_2sig[:, 0], PlanckTT_lowP_2sig[:, 1], 'r-', label='PlanckTT+lowP')
#ax.plot(PlanckTT_lowP_BKP_2sig[:, 0], PlanckTT_lowP_BKP_2sig[:, 1], 'g-', label='PlanckTT+lowP+BKP')
#ax.plot(lensing_ext_2sig[:, 0], lensing_ext_2sig[:, 1], 'b-', label='+lensing+ext')
#ax.fill_between(PlanckTT_lowP_1sig[:, 0], 0, PlanckTT_lowP_1sig[:, 1], alpha=0.5, facecolor='red')
#ax.fill_between(PlanckTT_lowP_2sig[:, 0], 0, PlanckTT_lowP_2sig[:, 1], alpha=0.5, facecolor='red')
#ax.fill_between(PlanckTT_lowP_BKP_1sig[:, 0], 0, PlanckTT_lowP_BKP_1sig[:, 1], alpha=0.5, facecolor='green')
#ax.fill_between(PlanckTT_lowP_BKP_2sig[:, 0], 0, PlanckTT_lowP_BKP_2sig[:, 1], alpha=0.5, facecolor='green')
#ax.fill_between(lensing_ext_1sig[:, 0], 0, lensing_ext_1sig[:, 1], alpha=0.5, facecolor='blue')
#ax.fill_between(lensing_ext_2sig[:, 0], 0, lensing_ext_2sig[:, 1], alpha=0.5, facecolor='blue')
#ax.set_xlim([0.945, 1.0])
#ax.set_ylim([0.0, 0.26])
#ax.set_title('\n'.join(wrap(r'A plot of the tensor to scalar ratio, $r_{0.002}$ against the spectral index, $n_{s}$.', 50)))
#ax.set_xlabel(r'$n_{s}$')
#ax.set_ylabel(r'$r_{0.002}$')
#ax.legend(prop={'size':10}, loc="upper right")
#ax.set_aspect(1./ax.get_data_ratio())
#plt.savefig("ns_r.pdf", dpi=2000, bbox_inches=0)

#############################################################################################################################

### Print out time taken to run program. ###
print("--- %s seconds ---" % (time.time() - start_time))
