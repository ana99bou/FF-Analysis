### Current error. mass has NaN-> cosh for vaslues lower than one

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cmath
import pandas as pd
from scipy.optimize import minimize
import sys 
import os
import scipy
import Ensemble as Ens
import argparse
from iminuit import Minuit


def pvalue(chi2, dof):
    r"""Compute the $p$-value corresponding to a $\chi^2$ with `dof` degrees
    of freedom."""
    return 1 - scipy.stats.chi2.cdf(chi2, dof)


def exp_val(data):
    return sum(data)/len(data)

def jack(x,j):
    r=0
    for i in range(len(x)):
        if i!=j:
            r=r+x[i]
    return 1/(len(x)-1)*r        


def var(data):
    res=0
    for i in range(len(data)):
        res=res+(jack(data,i)-exp_val(data))**2
    return np.sqrt((len(data)-1)/len(data)*res)

def extract(lst,number):
    return [item[number] for item in lst]


Ensemble = 'F1S'  # Example value, replace with actual input
particle = 'Ds'  # Example value, replace with actual input
nsq = 0  # Example value, replace with actual input
cmass_index = 0  # Example value, replace with actual input



if Ensemble == 'F1S':
    reg_low=19#18
    reg_up=25
elif Ensemble in ['M1', 'M2', 'M3']:
    reg_low=19#17#12
    reg_up=25
elif Ensemble in ['C1', 'C2']:
    reg_low=14#10
    reg_up=25

reg_up=reg_up+1

cmass=Ens.getCmass(Ensemble)[cmass_index]

#Get Path
fpath='../Data/{}/BsDsStar_{}_2pt{}.h5'.format(Ensemble,Ensemble,particle)
f=h5py.File(fpath, "r")
path='../Data/{}/2pt/'.format(Ensemble,Ensemble,particle)
configs,dt,ti,L= Ens.getEns(Ensemble)
m,csw,zeta=Ens.getRHQparams(Ensemble)
sm=Ens.getSM(Ensemble)
smass=Ens.getSmass(Ensemble)


def get_dataset(f, particle, style, sm, smass, cmass, nsq, m, csw, zeta):
    """
    style: 'old' or 'new'
    particle: 'Ds' or 'Bs'
    Returns the raw h5py dataset handles.
    """
    if particle == "Bs":
        if style == "old":
            path = f"/hl_SM{sm}_PT_{smass}_m{m}_csw{csw}_zeta{zeta}/operator_Gamma5/n2_0/data"
        elif style == "new":
            path = f"/hl_SM{sm}_SM{sm}_{smass}_m{m}_csw{csw}_zeta{zeta}/operator_Gamma5/n2_0/data"
        else:
            raise ValueError("Unknown style")
        return f[path]

    elif particle == "Ds":
        if style == "old":
            base = f"/cl_SM{sm}_PT_{smass}/c{cmass}"
        elif style == "new":
            base = f"/cl_SM{sm}_SM{sm}_{smass}/c{cmass}"
        else:
            raise ValueError("Unknown style")

        dsx = f[f"{base}/operator_GammaX/n2_{nsq}/data"]
        dsy = f[f"{base}/operator_GammaY/n2_{nsq}/data"]
        dsz = f[f"{base}/operator_GammaZ/n2_{nsq}/data"]
        return dsx, dsy, dsz

    else:
        raise ValueError("Unknown particle type")


def build_mirrored_correlator(data, particle, configs, ti):
    """
    Builds the mirrored (folded) correlator array of shape (configs, ti/2+1).
    `data`: either a single dataset (Bs) or a tuple of three datasets (Ds).
    """
    nt_half = int(ti // 2)

    if particle == "Bs":
        bs = data
        mir = np.zeros((configs, nt_half + 1))
        for k in range(configs):
            # t=0 separately
            mir[k, 0] = np.real(np.mean(bs[k][:][0]))
            for j in range(nt_half):
                mir[k, j + 1] = (
                    np.mean(np.real(bs), axis=1)[k, j + 1]
                    + np.mean(np.real(bs), axis=1)[k, ti - 1 - j]
                ) / 2

    elif particle == "Ds":
        dsx, dsy, dsz = data
        mir = np.zeros((configs, nt_half + 1))
        for k in range(configs):
            # t=0 average over X,Y,Z
            mir[k, 0] = (
                np.real(np.mean(dsx[k][:][0]))
                + np.real(np.mean(dsy[k][:][0]))
                + np.real(np.mean(dsz[k][:][0]))
            ) / 3
            for j in range(nt_half):
                valx = (
                    np.mean(np.real(dsx), axis=1)[k, j + 1]
                    + np.mean(np.real(dsx), axis=1)[k, ti - 1 - j]
                ) / 2
                valy = (
                    np.mean(np.real(dsy), axis=1)[k, j + 1]
                    + np.mean(np.real(dsy), axis=1)[k, ti - 1 - j]
                ) / 2
                valz = (
                    np.mean(np.real(dsz), axis=1)[k, j + 1]
                    + np.mean(np.real(dsz), axis=1)[k, ti - 1 - j]
                ) / 2
                mir[k, j + 1] = (valx + valy + valz) / 3

    else:
        raise ValueError("Unknown particle")

    return mir

# Get old & new datasets
if particle == "Bs":
    bs_old = get_dataset(f, "Bs", "old", sm, smass, cmass, nsq, m, csw, zeta)
    bs_new = get_dataset(f, "Bs", "new", sm, smass, cmass, nsq, m, csw, zeta)

    mir_old = build_mirrored_correlator(bs_old, "Bs", configs, ti)
    mir_new = build_mirrored_correlator(bs_new, "Bs", configs, ti)

else:  # Ds
    ds_old = get_dataset(f, "Ds", "old", sm, smass, cmass, nsq, m, csw, zeta)
    ds_new = get_dataset(f, "Ds", "new", sm, smass, cmass, nsq, m, csw, zeta)

    mir_old = build_mirrored_correlator(ds_old, "Ds", configs, ti)
    mir_new = build_mirrored_correlator(ds_new, "Ds", configs, ti)

def build_correlator_and_error(mir, configs, ti):
    """
    From mirrored correlator (configs, ti/2+1) → (res, error, mir^T).
    """
    nt_half = ti // 2 + 1
    res = np.zeros(nt_half)
    err = np.zeros(nt_half)
    mirtr = mir.T

    def exp_val(x):
        return np.mean(x)

    def jack(x, j):
        return (np.sum(np.delete(x, j)) / (len(x) - 1))

    for t in range(nt_half):
        res[t] = exp_val(mir[:, t])
        r = 0
        for i in range(configs):
            r += (jack(mir[:, t], i) - res[t]) ** 2
        err[t] = np.sqrt((configs - 1) / configs * r)

    return res, err, mirtr


# --- Build for both datasets
res_old, err_old, mirtr_old = build_correlator_and_error(mir_old, configs, ti)
res_new, err_new, mirtr_new = build_correlator_and_error(mir_new, configs, ti)

# --- Overlay plot of old vs new
plt.figure(figsize=(8,6))

t_range = range(len(res_old))  # same length for old and new

plt.errorbar(t_range, res_old, yerr=err_old,
             fmt='x', capsize=3, label='Old dataset', color='blue')
plt.errorbar(t_range, res_new, yerr=err_new,
             fmt='x', capsize=3, label='New dataset', color='orange')

plt.yscale('log')
plt.xlabel('Time slice $t$')
plt.ylabel('Correlator (log scale)')
plt.title(f'{particle} Correlator Comparison')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.savefig('Temp.pdf')


###############################################################################

def build_joint_covariance(mir_old, mir_new, res_old, res_new, reg_low, reg_up, configs):
    """
    Build joint covariance matrix for old+new correlators in [reg_low, reg_up).
    mir_old/new: transposed jackknife samples (time, configs)
    res_old/new: mean correlators
    """
    Nt = reg_up - reg_low
    dim = 2 * Nt
    cov = np.zeros((dim, dim))

    # helper: jackknife sample for one timeslice
    def jack(arr, j):
        return (np.sum(np.delete(arr, j)) / (len(arr) - 1))

    for i in range(Nt):
        for j in range(Nt):
            x = 0
            y = 0
            z = 0
            w = 0
            for k in range(configs):
                # old-old
                x += (jack(mir_old[reg_low+i], k) - res_old[reg_low+i]) * \
                     (jack(mir_old[reg_low+j], k) - res_old[reg_low+j])
                # old-new
                y += (jack(mir_old[reg_low+i], k) - res_old[reg_low+i]) * \
                     (jack(mir_new[reg_low+j], k) - res_new[reg_low+j])
                # new-old
                z += (jack(mir_new[reg_low+i], k) - res_new[reg_low+i]) * \
                     (jack(mir_old[reg_low+j], k) - res_old[reg_low+j])
                # new-new
                w += (jack(mir_new[reg_low+i], k) - res_new[reg_low+i]) * \
                     (jack(mir_new[reg_low+j], k) - res_new[reg_low+j])

            cov[i, j] = (configs-1)/configs * x
            cov[i, Nt+j] = (configs-1)/configs * y
            cov[Nt+i, j] = (configs-1)/configs * z
            cov[Nt+i, Nt+j] = (configs-1)/configs * w

    return cov

def chi2_joint(A_old, A_new, m, res_old, res_new, cov_inv, reg_low, reg_up):
    Nt = reg_up - reg_low
    residuals = []

    for i in range(reg_low, reg_up):
        residuals.append(res_old[i] - A_old * np.exp(-m * i))
    for i in range(reg_low, reg_up):
        residuals.append(res_new[i] - A_new * np.exp(-m * i))

    residuals = np.array(residuals)
    return residuals @ cov_inv @ residuals


# Build joint covariance
cov_joint = build_joint_covariance(mirtr_old, mirtr_new, res_old, res_new, reg_low, reg_up, configs)
cov_inv = np.linalg.inv(cov_joint)

# Initial guesses
#A0_old = res_old[reg_low]
#A0_new = res_new[reg_low]
m0 = np.log(res_old[reg_low]/res_old[reg_low+1])  # rough log-ratio
A0_old=res_old[reg_low]*np.exp(m0*reg_low)
A0_new=res_new[reg_low]*np.exp(m0*reg_low)

# Fit
res_fit = minimize(lambda p: chi2_joint(p[0], p[1], p[2],
                                        res_old, res_new,
                                        cov_inv, reg_low, reg_up),
                   x0=[A0_old, A0_new, m0], method='Nelder-Mead')

A_old_best, A_new_best, m_best = res_fit.x
chi2_min = res_fit.fun


print("Best fit parameters:")
print("A_old =", A_old_best)
print("A_new =", A_new_best)
print("A_old sqrt =", np.sqrt(A_new_best))
print("m     =", m_best)
print("chi²  =", chi2_min)





# Time range (full or just half)
t_range = np.arange(len(res_old))

# Fitted curves
fit_old = [A_old_best * np.exp(-m_best * t) for t in t_range]
fit_new = [A_new_best * np.exp(-m_best * t) for t in t_range]

plt.figure(figsize=(8,6))

# Data points
plt.errorbar(t_range, res_old, yerr=err_old, fmt='x', capsize=3,
             label='Old dataset', color='blue')
plt.errorbar(t_range, res_new, yerr=err_new, fmt='x', capsize=3,
             label='New dataset', color='orange')

# Fit curves
plt.plot(t_range, fit_old, '-', color='blue', label='Fit (old)')
plt.plot(t_range, fit_new, '-', color='orange', label='Fit (new)')

# Log scale and cosmetics
plt.yscale('log')
#plt.xlabel('Time slice $t$')
#plt.ylabel('Correlator (log scale)')
#plt.title(f'{particle} Correlator Combined Fit')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("CorrelatorCombinedFit.pdf")
plt.show()




'''
#Covarianze matrix 
masstmp=np.log(res[reg_low]/res[reg_low+1])
#masstmp=0.7341573429107688
print(masstmp)
a0tmp=res[reg_low]/(np.exp(-masstmp*reg_low)+np.exp(-masstmp*(ti/2+1-reg_low)))
#a0tmp=res[20]/(np.exp(-masstmp*20)+np.exp(-masstmp*(ti/2+1-20)))#-0.0000562341e-14

covmat=np.zeros(shape=(int(reg_up-reg_low),int(reg_up-reg_low)))
for t1 in range(int(reg_up-reg_low)):
    for t2 in range(int(reg_up-reg_low)):
        x=0
        for i in range(configs):  
            x=x+(jack(mirtr[t1+reg_low],i)-res[t1+reg_low])*(jack(mirtr[t2+reg_low],i)-res[t2+reg_low])
        covmat[t1][t2]=(configs-1)/configs*x
        covmat[t2][t1]=(configs-1)/configs*x  

invcovmat=np.linalg.inv(covmat) 
lst=list(range(reg_low, reg_up))



def chi(a1, a2):
    return np.dot(
        np.transpose([res[i] - (a1 * (np.exp(-a2 * i) + np.exp(-a2 * (ti - i)))) for i in lst]),
        np.matmul(invcovmat, [res[i] - (a1 * (np.exp(-a2 * i) + np.exp(-a2 * (ti - i)))) for i in lst])
    )
mbar = Minuit(chi, a1=1, a2=1)
mbar.errordef = 1  # For least-squares (χ²), errordef = 1
mbar.migrad()      # Run the minimization


print('Initial guess: a1 = {}, a2 = {}'.format(a0tmp, masstmp))
print("Optimized a1:", mbar.values["a1"])
print("Optimized a2 (mass):", mbar.values["a2"])
print("Minimum chi²:", mbar.fval)

best_a1 = mbar.values["a1"]
best_a2 = mbar.values["a2"]

def jackmass(t1,i):
    return jack(mirtr[t1],i)

def chijack(a1, a2, k):
    return np.dot(
        np.transpose([jackmass(i, k) - (a1 * (np.exp(-a2 * i) + np.exp(-a2 * (ti - i)))) for i in lst]),
        np.matmul(invcovmat, [jackmass(i, k) - (a1 * (np.exp(-a2 * i) + np.exp(-a2 * (ti - i)))) for i in lst])
    )


#Std Deviation for all jakcknife blocks
jblocks=np.zeros(configs)
h=0

for i in range(configs):
    mj = Minuit(lambda a1, a2: chijack(a1, a2, i), a1=a0tmp, a2=masstmp)
    mj.errordef = 1
    mj.migrad()
    jblocks[i] = mj.values["a2"]
    h += (mj.values["a2"] - best_a2) ** 2
sigma = np.sqrt((configs - 1) / configs * h)

print('Fehler:',sigma)

plt.figure(figsize=(8,6))

# Full time range
full_range = list(range(ti // 2 + 1))

# Plot all correlator points with errors
plt.errorbar(full_range, [res[t] for t in full_range], yerr=[error[t] for t in full_range], fmt='o', label='Correlator', capsize=3)

# Full fit function over the whole range
full_fit = [best_a1 * (np.exp(-best_a2 * t) + np.exp(-best_a2 * (ti - t))) for t in full_range]
plt.plot(full_range, full_fit, color='red', label='Fit: $A_1 e^{-A_2 t} + A_1 e^{-A_2 (T - t)}$')

plt.yscale('log')  
plt.xlabel('Time Slice $t$')
plt.ylabel('Correlator (log scale)')
plt.title(f'{particle} Correlator Fit')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('Test-Fig.pdf')
'''