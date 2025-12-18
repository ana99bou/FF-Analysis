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
cmass_index = 1  # Example value, replace with actual input

'''
parser = argparse.ArgumentParser()
parser.add_argument('--ensemble', type=str, required=True)
parser.add_argument('--particle', type=str, required=True)
parser.add_argument('--nsq', type=int, required=True)
parser.add_argument('--cmass_index', type=int, required=True)
args = parser.parse_args()

# Use the parsed arguments
Ensemble = args.ensemble
particle = args.particle
nsq = args.nsq
cmass_index = args.cmass_index
'''


if Ensemble == 'F1S':
    reg_low=10
    reg_up=25
elif Ensemble in ['M1', 'M2', 'M3']:
    reg_low=9
    reg_up=25
elif Ensemble in ['C1', 'C2']:
    reg_low=5
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


###############################################################################

# --- Frozen covariance (central fit) ---
def build_joint_covariance(mir_old, mir_new, res_old, res_new, reg_low, reg_up, configs):
    Nt = reg_up - reg_low
    dim = 2 * Nt
    cov = np.zeros((dim, dim))

    def jack(arr, j):
        return (np.sum(np.delete(arr, j)) / (len(arr) - 1))

    for i in range(Nt):
        for j in range(Nt):
            x = y = z = w = 0
            for k in range(configs):
                x += (jack(mir_old[reg_low+i], k) - res_old[reg_low+i]) * \
                     (jack(mir_old[reg_low+j], k) - res_old[reg_low+j])
                y += (jack(mir_old[reg_low+i], k) - res_old[reg_low+i]) * \
                     (jack(mir_new[reg_low+j], k) - res_new[reg_low+j])
                z += (jack(mir_new[reg_low+i], k) - res_new[reg_low+i]) * \
                     (jack(mir_old[reg_low+j], k) - res_old[reg_low+j])
                w += (jack(mir_new[reg_low+i], k) - res_new[reg_low+i]) * \
                     (jack(mir_new[reg_low+j], k) - res_new[reg_low+j])

            cov[i, j]         = (configs-1)/configs * x
            cov[i, Nt+j]      = (configs-1)/configs * y
            cov[Nt+i, j]      = (configs-1)/configs * z
            cov[Nt+i, Nt+j]   = (configs-1)/configs * w

    cov = 0.5*(cov+cov.T)
    return cov

cov_joint = build_joint_covariance(mirtr_old, mirtr_new, res_old, res_new, reg_low, reg_up, configs)
cov_inv = np.linalg.inv(cov_joint)

# --- Unfrozen covariance (per block jackknife) ---
def build_joint_covariance_unfrozen_block(mirtr_old, mirtr_new, omit_idx, reg_low, reg_up):
    keep = [i for i in range(mirtr_old.shape[1]) if i != omit_idx]
    n = len(keep)
    Nt = reg_up - reg_low

    D_old = mirtr_old[reg_low:reg_up][:, keep]
    D_new = mirtr_new[reg_low:reg_up][:, keep]

    mu_old = D_old.mean(axis=1, keepdims=True)
    mu_new = D_new.mean(axis=1, keepdims=True)

    U_old = D_old - mu_old
    U_new = D_new - mu_new

    S_old = (U_old @ U_old.T) / max(n-1, 1)
    S_new = (U_new @ U_new.T) / max(n-1, 1)
    S_on  = (U_old @ U_new.T) / max(n-1, 1)

    C_old = S_old / max(n, 1)
    C_new = S_new / max(n, 1)
    C_on  = S_on  / max(n, 1)

    cov = np.block([[C_old, C_on],[C_on.T, C_new]])
    cov = 0.5*(cov+cov.T)

    eps = 1e-12
    diag_scale = np.trace(cov)/cov.shape[0] if cov.shape[0] > 0 else 1.0
    cov += eps*diag_scale*np.eye(cov.shape[0])

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov, rcond=1e-12)

    return cov, cov_inv


# Build joint covariance
cov_joint = build_joint_covariance(mirtr_old, mirtr_new, res_old, res_new, reg_low, reg_up, configs)
cov_inv = np.linalg.inv(cov_joint)


def fitfunc(t, A0, A1, m0, m1):
    return A0 * np.exp(-m0 * t) + A1 * np.exp(-m1 * t)

def chi2_two_state(params):
    A0_old, A1_old, A0_new, A1_new, m0, m1 = params
    
    model_old = np.array([fitfunc(t, A0_old, A1_old, m0, m1) for t in t_range])
    model_new = np.array([fitfunc(t, A0_new, A1_new, m0, m1) for t in t_range])
    
    resid = np.concatenate([
        res_old[reg_low:reg_up] - model_old[reg_low:reg_up],
        res_new[reg_low:reg_up] - model_new[reg_low:reg_up]
    ])
    
    return resid @ cov_inv @ resid


# Initial guesses
m0_init = np.log(res_old[reg_low+5]/res_old[reg_low+6])  # rough log-ratio
A0_old_init=res_old[reg_low]*np.exp(m0_init*reg_low)
A0_new_init=res_new[reg_low]*np.exp(m0_init*reg_low)
p0 = [A0_old_init, A0_old_init, A0_new_init, A0_new_init, m0_init, m0_init+0.2]  # initial guesses: amplitudes, m0, m1
p0= A0_old_init, A0_old_init, A0_new_init, A0_new_init, m0_init, m0_init+0.2
#res_fit = minimize(chi2_two_state, p0, method='BFGS')
#A0_old, A1_old, A0_new, A1_new, m0_best, m1_best = res_fit.x

print(p0)

# ------------------------ Remaining code: Minuit + jackknife ------------------------

tvec = np.arange(reg_low, reg_up)

# chi2 with constraint m1 > m0 implemented via reparametrization
def chi2_minuit(A0_old, A1_old, A0_new, A1_new, m0, dm):
    m1 = m0 + dm  # enforce m1 > m0
    model_old = fitfunc(tvec, A0_old, A1_old, m0, m1)
    model_new = fitfunc(tvec, A0_new, A1_new, m0, m1)

    resid = np.concatenate([
        res_old[reg_low:reg_up] - model_old,
        res_new[reg_low:reg_up] - model_new
    ])

    return float(resid @ cov_inv @ resid)


def run_minuit(initial_params, limits=None, fixed=None):
    m = Minuit(chi2_minuit, **initial_params)
    for name in initial_params.keys():
        m.errors[name] = abs(initial_params[name]) * 0.01 if initial_params[name] != 0 else 0.01
    if limits:
        for name, lim in limits.items():
            m.limits[name] = lim
    if fixed:
        for name in fixed:
            m.fixed[name] = True
    try:
        m.migrad()
        m.hesse()
    except Exception as e:
        print("Minuit failed with:", e)
    return m


# Initial parameter guesses
try:
    p0
except NameError:
    m0_init = max(0.05, np.log(res_old[reg_low+5]/res_old[reg_low+6]))
    A0_old_init = res_old[reg_low]*np.exp(m0_init*reg_low)
    A0_new_init = res_new[reg_low]*np.exp(m0_init*reg_low)
    p0 = [A0_old_init, A0_old_init, A0_new_init, A0_new_init, m0_init, 0.2]

initial_params = {
    'A0_old': p0[0],
    'A1_old': p0[1],
    'A0_new': p0[2],
    'A1_new': p0[3],
    'm0':     p0[4],
    'dm':     0.2 #abs(p0[5] - p0[4])  # positive gap
}

limits = {
    'm0': (1e-6, None),
    'dm': (1e-6, None)
}

# Central fit
m_central = run_minuit(initial_params, limits=limits)
central_vals = m_central.values.to_dict()
hesse_errors = m_central.errors.to_dict()

# translate back to (m0, m1)
m0_best = central_vals['m0']
m1_best = m0_best + central_vals['dm']
chi2_central = float(m_central.fval)
ndof = 2*(reg_up-reg_low) - len(central_vals)

print("\n=== Central fit (Minuit, with m1>m0) ===")
for k in ['A0_old','A1_old','A0_new','A1_new']:
    print(f"{k:8s} = {central_vals[k]:12.6e} (err: {hesse_errors.get(k, np.nan):.3e})")
print(f"m0       = {m0_best:12.6e} (err: {hesse_errors.get('m0', np.nan):.3e})")
print(f"m1       = {m1_best:12.6e} (err: {hesse_errors.get('dm', np.nan):.3e})")
print(f"chi2 = {chi2_central:.6f}, dof = {ndof}, p-value = {pvalue(chi2_central, ndof):.4f}")

# ------------------------ Jackknife errors only ------------------------

def jack_mean_timeseries(mirtr, omit_index):
    """
    Compute jackknife average correlator (time series) leaving out config omit_index.
    mirtr: array (time, configs)
    """
    nt = mirtr.shape[0]
    out = np.empty(nt)
    for t in range(nt):
        out[t] = (np.sum(mirtr[t]) - mirtr[t, omit_index]) / (mirtr.shape[1] - 1)
    return out

param_names = list(initial_params.keys())
jack_params = np.zeros((configs, len(param_names)))

for k in range(configs):
    # leave-one-out mean correlators
    ro = jack_mean_timeseries(mirtr_old, k)
    rn = jack_mean_timeseries(mirtr_new, k)

    # leave-one-out covariance
    cov_jk, cov_inv_jk = build_joint_covariance_unfrozen_block(
        mirtr_old, mirtr_new, k, reg_low, reg_up
    )

    # chi² for block k
    def chi2_jk(A0_old, A1_old, A0_new, A1_new, m0, dm):
        m1 = m0 + dm
        model_old = fitfunc(tvec, A0_old, A1_old, m0, m1)
        model_new = fitfunc(tvec, A0_new, A1_new, m0, m1)
        resid = np.concatenate([ro[reg_low:reg_up] - model_old,
                                rn[reg_low:reg_up] - model_new])
        return float(resid @ cov_inv_jk @ resid)

    m_jk = Minuit(chi2_jk, **central_vals)
    for name in central_vals:
        m_jk.errors[name] = max(abs(central_vals[name])*0.01, 1e-6)
    for name, lim in limits.items():
        m_jk.limits[name] = lim

    try:
        m_jk.migrad()
        vals = [m_jk.values[name] for name in param_names]
    except Exception as e:
        print(f"Minuit failed on jackknife sample {k}: {e}")
        vals = [central_vals[name] for name in param_names]

    jack_params[k, :] = vals

# --- Jackknife errors relative to central fit ---
fac = (configs - 1) / configs
jack_var = fac * np.sum((jack_params - np.array([central_vals[n] for n in param_names]))**2, axis=0)
jack_err = np.sqrt(jack_var)

print("\n=== Final results (central from full fit, errors from jackknife) ===")
for i, name in enumerate(param_names):
    if name == 'dm':
        print(f"m1-m0 (dm) = {central_vals['dm']:12.6e} ± {jack_err[i]:.3e}")
    else:
        print(f"{name:8s} = {central_vals[name]:12.6e} ± {jack_err[i]:.3e}")

# compute jackknife m1 values directly
m0_jack = jack_params[:, param_names.index("m0")]
dm_jack = jack_params[:, param_names.index("dm")]
m1_jack = m0_jack + dm_jack

# central values
m0_central = central_vals['m0']
m1_central = central_vals['m0'] + central_vals['dm']

# jackknife error for m1
fac = (configs - 1) / configs
m1_var = fac * np.sum((m1_jack - m1_central)**2)
m1_err = np.sqrt(m1_var)

# print results
print(f"\nm0 = {m0_central:.6e} ± {jack_err[param_names.index('m0')]:.3e}")
print(f"m1 = {m1_central:.6e} ± {m1_err:.3e}")

# --- prepare fit curves ---
t_fit = np.arange(reg_low, reg_up+1)

fit_old = fitfunc(
    t_fit,
    central_vals['A0_old'], central_vals['A1_old'],
    central_vals['m0'], central_vals['m0']+central_vals['dm']
)

fit_new = fitfunc(
    t_fit,
    central_vals['A0_new'], central_vals['A1_new'],
    central_vals['m0'], central_vals['m0']+central_vals['dm']
)

'''
plt.figure(figsize=(8,6))

# full time range for data
t_all = np.arange(len(res_old))

# plot data
plt.errorbar(t_all, res_old, yerr=err_old, fmt='o', color='red', label="Old data")
plt.errorbar(t_all, res_new, yerr=err_new, fmt='o', color='blue', label="New data")

# fit only within [reg_low, reg_up)
t_fit = np.arange(reg_low, reg_up)

fit_old = fitfunc(t_fit, central_vals["A0_old"], central_vals["A1_old"],
                  central_vals["m0"], central_vals["m0"] + central_vals["dm"])
fit_new = fitfunc(t_fit, central_vals["A0_new"], central_vals["A1_new"],
                  central_vals["m0"], central_vals["m0"] + central_vals["dm"])

plt.plot(t_fit, fit_old, '-', color='darkred', label="Fit (old)")
plt.plot(t_fit, fit_new, '-', color='darkblue', label="Fit (new)")

plt.yscale("log")
plt.xlabel("Time slice $t$")
plt.ylabel("Correlator")
plt.title(f"{particle} two-state fit, ensemble {Ensemble}")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("fit_result.pdf")
plt.show()
'''

print(jack_err)

df1 = pd.DataFrame([{
    'A0_old': central_vals['A0_old'],
    'DeltaA0_old': jack_err[0],
    'A1_old': central_vals['A1_old'],
    'DeltaA1_old': jack_err[1],
    'A0_new': central_vals['A0_new'],
    'DeltaA0_new': jack_err[2],
    'A1_new': central_vals['A1_new'],
    'DeltaA1_new': jack_err[3],
    'Mass0': central_vals['m0'],
    'DeltaM0': jack_err[4],
    'Mass1': central_vals['m0']+central_vals['dm'],
    'DeltaM1': m1_err,
    'DeltaM': central_vals['dm'],
    'DeltaDm': jack_err[5],
    'RegUp': reg_up,
    'RegLow': reg_low
}])

df2 = pd.DataFrame([{
    'chi2': chi2_central,
    'dof': ndof,
    'p-val': pvalue(chi2_central, ndof)
}])

df1.to_csv(path+'Excited-comb-unfrozen-Ds{}Result-{}.csv'.format(cmass,nsq), sep='\t')
df2.to_csv(path+'Excited-comb-unfrozen-pval-Ds{}-{}.csv'.format(cmass,nsq), sep='\t')


# --- build DataFrame of jackknife blocks ---
idx_m0 = param_names.index("m0")
idx_dm = param_names.index("dm")

# compute m1 for each jackknife sample
m1_jack = jack_params[:, idx_m0] + jack_params[:, idx_dm]

# combine into DataFrame
df = pd.DataFrame({
    "A0_old": jack_params[:, param_names.index("A0_old")],
    "A1_old": jack_params[:, param_names.index("A1_old")],
    "A0_new": jack_params[:, param_names.index("A0_new")],
    "A1_new": jack_params[:, param_names.index("A1_new")],
    "m0":     jack_params[:, idx_m0],
    "m1":     m1_jack,
    "dm":     jack_params[:, idx_dm],
})

# --- save path ---
outdir = '../Data/{}/2pt/'.format(Ensemble)
os.makedirs(outdir, exist_ok=True)

outfile = outdir + 'Excited-comb-unfrozen-Blocks-Ds{}-{}.csv'.format(cmass, nsq)
df.to_csv(outfile, index=False)

print(f"Jackknife block values saved to: {outfile}")


def effmass(corr):
    """Compute effective mass from correlator array corr[t]."""
    Nt = len(corr)
    out = np.zeros(Nt-2)  # defined for t=0..Nt-3
    for t in range(Nt-2):
        ratio = (corr[t] + corr[t+2]) / (2.0 * corr[t+1])
        out[t] = np.arccosh(ratio)
    return out

def jackknife_effmass(mirtr, configs):
    """
    Compute jackknife errors for effective mass.
    mirtr: shape (time, configs) (transposed jackknife correlator samples)
    Returns central values and errors.
    """
    Nt = mirtr.shape[0]
    # central correlator = mean over configs
    res = np.mean(mirtr, axis=1)
    eff_central = effmass(res)

    # jackknife samples
    nblocks = configs
    eff_jack = np.zeros((nblocks, Nt-2))
    for k in range(nblocks):
        corr_jk = (np.sum(mirtr, axis=1) - mirtr[:,k]) / (configs-1)
        eff_jack[k,:] = effmass(corr_jk)

    # jackknife variance
    fac = (configs-1)/configs
    diffs = eff_jack - eff_central[None,:]
    err = np.sqrt(fac * np.sum(diffs**2, axis=0))
    return eff_central, err

# --- effective mass for old and new datasets ---
eff_old, err_eff_old = jackknife_effmass(mirtr_old, configs)
eff_new, err_eff_new = jackknife_effmass(mirtr_new, configs)


def effmass_from_params(A0, A1, m0, m1, Nt):
    corr = np.array([A0*np.exp(-m0*t) + A1*np.exp(-m1*t) for t in range(Nt)])
    return effmass(corr)

Nt = len(res_old)
t_eff = np.arange(Nt-2)

# --- central fit curves ---
eff_fit_old = effmass_from_params(
    central_vals['A0_old'], central_vals['A1_old'],
    central_vals['m0'], central_vals['m0']+central_vals['dm'], Nt
)
eff_fit_new = effmass_from_params(
    central_vals['A0_new'], central_vals['A1_new'],
    central_vals['m0'], central_vals['m0']+central_vals['dm'], Nt
)

# --- jackknife error bands ---
idx_m0 = param_names.index("m0")
idx_dm = param_names.index("dm")

eff_fit_old_jack = np.zeros((configs, Nt-2))
eff_fit_new_jack = np.zeros((configs, Nt-2))

for k in range(configs):
    vals = {name: jack_params[k,i] for i,name in enumerate(param_names)}
    m0_k = vals['m0']
    m1_k = vals['m0'] + vals['dm']

    eff_fit_old_jack[k,:] = effmass_from_params(
        vals['A0_old'], vals['A1_old'], m0_k, m1_k, Nt
    )
    eff_fit_new_jack[k,:] = effmass_from_params(
        vals['A0_new'], vals['A1_new'], m0_k, m1_k, Nt
    )

fac = (configs-1)/configs
err_fit_old = np.sqrt(fac * np.sum((eff_fit_old_jack - eff_fit_old[None,:])**2, axis=0))
err_fit_new = np.sqrt(fac * np.sum((eff_fit_new_jack - eff_fit_new[None,:])**2, axis=0))

# --- plotting ---
plt.figure(figsize=(8,6))

# full effective mass data
plt.errorbar(t_eff, eff_old, yerr=err_eff_old, fmt='o', color='red',
             capsize=3, label="PT_SM")
plt.errorbar(t_eff, eff_new, yerr=err_eff_new, fmt='o', color='blue',
             capsize=3, label="SM_SM")

# restrict fits to [reg_low, reg_up)
t_fit_eff = np.arange(reg_low, reg_up)

plt.plot(t_fit_eff, eff_fit_old[reg_low:reg_up], '-', color='darkred', lw=2)
plt.plot(t_fit_eff, eff_fit_new[reg_low:reg_up], '-', color='darkblue', lw=2)

plt.fill_between(t_fit_eff,
                 (eff_fit_old - err_fit_old)[reg_low:reg_up],
                 (eff_fit_old + err_fit_old)[reg_low:reg_up],
                 color='red', alpha=0.3)

plt.fill_between(t_fit_eff,
                 (eff_fit_new - err_fit_new)[reg_low:reg_up],
                 (eff_fit_new + err_fit_new)[reg_low:reg_up],
                 color='blue', alpha=0.3)

# mark fit window
plt.axvline(reg_low, color='gray', ls='--', lw=1, alpha=0.5)
plt.axvline(reg_up-1, color='gray', ls='--', lw=1, alpha=0.5)

plt.xlabel("Time", fontsize=16)
plt.ylabel(r"$E_{\rm eff}(t)$", fontsize=16)
#plt.title("Effective mass with two-state fit curves (fit window only)")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend(fontsize=14)
plt.ylim(0.65, 1.3)
plt.tight_layout()
plt.savefig(outdir+'Effective_mass_with_fit_band-unfrozen-{}-nsq{}.pdf'.format(cmass,nsq))
