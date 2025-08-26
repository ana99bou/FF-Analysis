import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import GetCombs
import Folding
import Ensemble as Ens
import Jackblocks
import Ratio
import Basic
import Regression
import scipy
import sys
 
#########Choose Params
'''
FF = sys.argv[1]
#nsq = int(sys.argv[2])
cmass_index = int(sys.argv[3])
ensemble = sys.argv[4]
use_disp=bool(int(sys.argv[5]))
frozen_analysis = bool(int(sys.argv[6]))
'''


FF='V'
#nsq=1
cmass_index=1
ensemble='F1S'
use_disp=True


#Ens.getCmass(ensemble) gives us an array of the different charm masses for each ens; chose which one
cmass=Ens.getCmass(ensemble)[cmass_index]

#Fit range
if ensemble == 'F1S':
    reg_up=25
    reg_low=15
elif ensemble in ['M1', 'M2', 'M3']:
    reg_up=22
    reg_low=18
elif ensemble in ['C1', 'C2']:
    reg_up=16
    reg_low=12

reg_up=reg_up+1

# Get strange mass and smearing radius
sm=Ens.getSM(ensemble)
smass=Ens.getSmass(ensemble)


# Ensemble details
inv=Ens.getInvSpac(ensemble)
nconf,dt,ts,L= Ens.getEns(ensemble)
m,csw,zeta=Ens.getRHQparams(ensemble)

# Read h5 file -> 
##################TODO three h5files for each 3pt and 2pt and get the naming consistent

'''
if ensemble == 'F1S':
    f3pt = h5py.File("../Data/{}/BsDsStar.h5".format(ensemble,ensemble), "r")
    f2ptDs = h5py.File("../Data/{}/BsDsStar.h5".format(ensemble,ensemble), "r")
    f2ptBs = h5py.File("../Data/{}/BsDsStar.h5".format(ensemble,ensemble), "r")
else:
'''
f3pt = h5py.File("../Data/{}/BsDsStar_{}_3pt.h5".format(ensemble,ensemble), "r")
f2ptDs = h5py.File("../Data/{}/BsDsStar_{}_2ptDs.h5".format(ensemble,ensemble), "r")
f2ptBs = h5py.File("../Data/{}/BsDsStar_{}_2ptBs.h5".format(ensemble,ensemble), "r")

# Instead of reading nsq from sys.argv
# Define which nsq values you want to process:
nsq_values = [1, 2, 3, 4, 5]  # adjust to your available data

# Storage for results
all_ratios = {}
all_errs = {}
all_jackknife_blocks = {}
all_avn0 = {}

for nsq in nsq_values:
    print(f"Processing nsq = {nsq} ...")

    # --- This is basically your original code from 'if use_disp' onwards ---
    if use_disp:
        md = pd.read_csv(f'../Data/{ensemble}/2pt/Ds{cmass}Result-0.csv', sep='\t', index_col=0).loc[0, 'EffectiveMass']

        vecs = [[0,0,0],[1,0,0],[1,1,0],[1,1,1],[2,0,0],[2,1,0]]
        def calculate_value(a, md, p, L):
            term1 = np.sinh(md / 2) ** 2
            term2 = np.sin(p[0] * np.pi / L) ** 2
            term3 = np.sin(p[1] * np.pi / L) ** 2
            term4 = np.sin(p[2] * np.pi / L) ** 2
            return 2 * np.arcsinh(np.sqrt(term1 + term2 + term3 + term4))

        ed = calculate_value(1/inv, md, vecs[nsq], L)

    else:
        edlist = [
            pd.read_csv(f'../Data/{ensemble}/2pt/Ds{cmass}Result-{i}.csv', sep='\t', index_col=0).loc[0,'EffectiveMass']
            for i in range(6)
        ]
        ed = edlist[nsq]
        md = edlist[0]
    mb=pd.read_csv('../Data/{}/2pt/BsResult.csv'.format(ensemble), sep='\t',index_col=0).loc[0,'EffectiveMass']

    # --- momentum combinations for this nsq ---
    if FF == 'V':
        mom, pref = GetCombs.get_moms_and_prefacs_V()
    elif FF == 'A0':
        mom, pref = GetCombs.get_moms_and_prefacs_A0()
    elif FF == 'A1':
        mom, pref = GetCombs.get_moms_and_prefacs_A1()
    elif FF == 'A2':
        mom, pref = GetCombs.get_moms_and_prefacs_A2()

    nmom = len(mom[nsq])

    # --- read datasets for this nsq ---
    dsets  = [f3pt[f"/CHARM_PT_SEQ_SM{sm}_s{smass}/c{cmass}/dT{dt}/{mom[nsq][i]}/forward/data"] for i in range(nmom)]
    dsetsb = [f3pt[f"/CHARM_PT_SEQ_SM{sm}_s{smass}/c{cmass}/dT{dt}/{mom[nsq][i]}/backward/data"] for i in range(nmom)]

    dsxn0 = f2ptDs[f"/cl_SM{sm}_SM{sm}_{smass}/c{cmass}/operator_GammaX/n2_{nsq}/data"]
    dsyn0 = f2ptDs[f"/cl_SM{sm}_SM{sm}_{smass}/c{cmass}/operator_GammaY/n2_{nsq}/data"]
    dszn0 = f2ptDs[f"/cl_SM{sm}_SM{sm}_{smass}/c{cmass}/operator_GammaZ/n2_{nsq}/data"]
    bsn0  = f2ptBs[f"/hl_SM{sm}_SM{sm}_{smass}_m{m}_csw{csw}_zeta{zeta}/operator_Gamma5/n2_0/data"]

    # read jackknife blocks for 2pt fits
    bsfit = pd.read_csv(f'../Data/{ensemble}/2pt/Bs-blocks.csv', sep='\s', engine="python")

    if use_disp:
        dsfit_0 = pd.read_csv(f'../Data/{ensemble}/2pt/Ds{cmass}-nsq0-blocks.csv', sep='\s', engine="python")
        md_values = dsfit_0['EffectiveMass']
        ed_jackknife = [calculate_value(1/inv, mdv, vecs[nsq], L) for mdv in md_values]
        dsfit = pd.DataFrame(ed_jackknife, columns=['EffectiveMass'])
    else:
        dsfit = pd.read_csv(f'../Data/{ensemble}/2pt/Ds{cmass}-nsq{nsq}-blocks.csv', sep='\s', engine="python")

    # --- folding ---
    if FF == 'V':
        pre = -(mb + md) / (2 * mb) * L / (2 * np.pi)
        av1n0 = Folding.folding3ptVec(dsets, dsetsb, nmom, dt, nconf, ts, pref, nsq)
    elif FF == 'A0':
        pre = md / (2 * ed * mb)
        av1n0 = Folding.folding3ptAx(dsets, dsetsb, nmom, dt, nconf, ts, pref, nsq)
    elif FF == 'A1':
        pre = 1 / (mb + ed)
        av1n0 = Folding.folding3ptAx(dsets, dsetsb, nmom, dt, nconf, ts, pref, nsq)
    elif FF == 'A2':
        pre = md**2 * (mb + md) / (ed * mb)
        av1n0 = Folding.folding3ptAxA2(dsets, dsetsb, nmom, dt, nconf, ts, pref, nsq)

    avdx = Folding.folding2pt3(dsxn0, dsyn0, dszn0, nmom, dt, nconf, ts)
    avb  = Folding.folding2pt(bsn0, nmom, dt, nconf, ts)

    if FF == 'A2':
        jb3pt = Jackblocks.create_blocks_3pt(av1n0, nmom, dt, nconf)
        jbdx  = Jackblocks.create_blocks_2pt(avdx, dt, nconf)
        jbb   = Jackblocks.create_blocks_2pt(avb, dt, nconf)
        ratiojack, errn0 = Ratio.build_Ratio_A2(jb3pt, jbdx, jbb, pref, dt, nsq, nconf, pre, md, mb, ed, dsfit, bsfit, L, A0fit, A1fit)
    else:
        jb3pt = Jackblocks.create_blocks_2pt(av1n0, dt, nconf)
        jbdx  = Jackblocks.create_blocks_2pt(avdx, dt, nconf)
        jbb   = Jackblocks.create_blocks_2pt(avb, dt, nconf)
        ratiojack, errn0 = Ratio.build_Ratio(jb3pt, jbdx, jbb, pref, dt, nsq, nconf, ed, mb, pre, dsfit, bsfit)

    avn0 = ratiojack[:, nconf]

    # --- store results ---
    all_ratios[nsq] = ratiojack
    all_errs[nsq] = errn0
    all_jackknife_blocks[nsq] = jb3pt
    all_avn0[nsq] = avn0

print("Finished reading and building ratios for all nsq.")

###############################################################################

def build_Covarianz_allnsq(all_ratios, reg_up, reg_low, nconf):
    """
    Build a single covariance matrix for all nsq values combined.
    all_ratios: dict {nsq: ratiojack array of shape (tmax, nconf+1)}
    reg_up/reg_low: fit range for time slices
    nconf: number of jackknife samples
    """
    # Make sure nsq order is consistent
    nsq_list = sorted(all_ratios.keys())

    # Extract and stack the fit windows for each nsq
    data_blocks = []
    for nsq in nsq_list:
        # For each jackknife block, take the fit range time slices
        # Shape (nconf, ntime_fit)
        block = all_ratios[nsq][reg_low:reg_up, :nconf].T  # transpose so shape is (nconf, ntime)
        data_blocks.append(block)

    # Stack along time dimension → shape (nconf, total_points)
    data_all = np.hstack(data_blocks)

    # Compute mean over jackknife samples
    mean_all = np.mean(data_all, axis=0)

    # Covariance (jackknife normalization factor)
    covmat = (nconf - 1) / nconf * np.dot(
        (data_all - mean_all).T,
        (data_all - mean_all)
    )

    return covmat, nsq_list

covmat_all, nsq_order = build_Covarianz_allnsq(all_ratios, reg_up, reg_low, nconf)

print("Combined covariance matrix shape:", covmat_all.shape)
print("nsq order in combined fit:", nsq_order)

# Optional: invert it for later fitting
invcovmat_all = np.linalg.inv(covmat_all)

decay_const = 0.2  # fixed total exponent

def model(params, tvals, nsq_order):
    n_nsq = len(nsq_order)
    A_vals = params[:n_nsq]
    B_vals = params[n_nsq:2*n_nsq]
    return np.concatenate([
        A_vals[i] + B_vals[i] * np.exp(-decay_const * tvals)
        for i in range(n_nsq)
    ])

def chi2(params, all_ratios, reg_low, reg_up, nsq_order, cov_inv):
    t_fit = np.arange(reg_low, reg_up)
    data = np.concatenate([
        np.mean(all_ratios[nsq][reg_low:reg_up, :], axis=1)
        for nsq in nsq_order
    ])
    diff = data - model(params, t_fit, nsq_order)
    return diff @ cov_inv @ diff

def jackknife_fit(all_ratios, reg_low, reg_up, nconf, frozen=True):
    nsq_order = sorted(all_ratios.keys())
    n_nsq = len(nsq_order)

    # jackknife samples in fit window
    jk_samples = []
    for i in range(nconf):
        jk_samples.append(np.concatenate([
            np.mean(np.delete(all_ratios[nsq], i, axis=1)[reg_low:reg_up, :], axis=1)
            for nsq in nsq_order
        ]))
    jk_samples = np.array(jk_samples)

    cov = np.cov(jk_samples, rowvar=False, ddof=1)
    cov_inv = np.linalg.inv(cov)

    # initial guess
    A0 = [np.mean(np.mean(all_ratios[nsq][reg_low:reg_up, :], axis=1)) for nsq in nsq_order]
    B0 = [0.1] * n_nsq
    p0 = A0 + B0

    # central fit
    result = minimize(
        chi2, p0, args=(all_ratios, reg_low, reg_up, nsq_order, cov_inv),
        method='BFGS'
    )
    central = result.x

    # jackknife fits
    jk_params = []
    for i in range(nconf):
        if not frozen:
            sub_jk = []
            for j in range(nconf):
                if j == i:
                    continue
                sub_jk.append(np.concatenate([
                    np.mean(np.delete(np.delete(all_ratios[nsq], i, axis=1), j, axis=1)[reg_low:reg_up, :], axis=1)
                    for nsq in nsq_order
                ]))
            cov_i = np.cov(np.array(sub_jk), rowvar=False, ddof=1)
            cov_inv_i = np.linalg.inv(cov_i)
        else:
            cov_inv_i = cov_inv

        res_i = minimize(
            chi2, p0,
            args=({nsq: np.delete(all_ratios[nsq], i, axis=1) for nsq in nsq_order},
                  reg_low, reg_up, nsq_order, cov_inv_i),
            method='BFGS'
        )
        jk_params.append(res_i.x)

    jk_params = np.array(jk_params)
    errs = np.sqrt((nconf - 1) * np.mean((jk_params - central) ** 2, axis=0))

    return central, errs, nsq_order, jk_params

def plot_all_t(all_ratios, nconf, nsq_order, central):
    plt.figure(figsize=(8,6))
    colors = plt.cm.tab10.colors
    n_nsq = len(nsq_order)
    A_vals = central[:n_nsq]
    B_vals = central[n_nsq:2*n_nsq]

    for i, nsq in enumerate(nsq_order):
        nt = all_ratios[nsq].shape[0]
        t_all = np.arange(nt)
        data_mean = np.mean(all_ratios[nsq][:, :nconf], axis=1)
        data_err = np.std(all_ratios[nsq][:, :nconf], axis=1, ddof=1) / np.sqrt(nconf)

        # data points
        plt.errorbar(t_all, data_mean, yerr=data_err, fmt='o',
                     color=colors[i % len(colors)], label=f'nsq={nsq}')

        # fit curve
        fit_curve = A_vals[i] + B_vals[i] * np.exp(-decay_const * t_all)
        plt.plot(t_all, fit_curve, '-', color=colors[i % len(colors)])

    plt.xlabel("t")
    plt.ylabel("Ratio")
    plt.title(f"Global fit: A_i + B_i * exp(-{decay_const} * t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('Test-Combined-new.png')

def plot_all_t_with_bands(all_ratios, nconf, nsq_order, central, jk_params):
    plt.figure(figsize=(8,6))
    colors = plt.cm.tab10.colors
    n_nsq = len(nsq_order)

    A_central = central[:n_nsq]
    B_central = central[n_nsq:2*n_nsq]

    # Split jk_params into A_jk and B_jk arrays for error bands
    jk_params = np.array(jk_params)
    A_jk = jk_params[:, :n_nsq]
    B_jk = jk_params[:, n_nsq:2*n_nsq]

    for i, nsq in enumerate(nsq_order):
        nt = all_ratios[nsq].shape[0]
        t_all = np.arange(nt)
        
        # Data mean & errors
        data_mean = all_avn0[nsq]#np.mean(all_ratios[nsq][:, :nconf], axis=1)
        data_err = all_errs[nsq] 

        plt.errorbar(t_all, data_mean, yerr=data_err, fmt='x',
                     color=colors[i % len(colors)], label=f'nsq={nsq}')

        # Central fit
        fit_central = A_central[i] + B_central[i] * np.exp(-decay_const * t_all)

        # Jackknife fit error band
        fit_jk_vals = []
        for k in range(nconf):
            A_k = A_jk[k, i]
            B_k = B_jk[k, i]
            fit_jk_vals.append(A_k + B_k * np.exp(-decay_const * t_all))
        fit_jk_vals = np.array(fit_jk_vals)

        fit_mean = np.mean(fit_jk_vals, axis=0)
        fit_err = np.sqrt((nconf - 1) * np.mean((fit_jk_vals - fit_mean)**2, axis=0))

        plt.plot(t_all, fit_central, '-', color=colors[i % len(colors)])
        plt.fill_between(t_all, fit_mean - fit_err, fit_mean + fit_err,
                         color=colors[i % len(colors)], alpha=0.3)

    plt.xlabel("t")
    plt.ylabel("Ratio")
    plt.title(f"Global fit: A_i + B_i * exp(-{decay_const} * t) with error bands")
    plt.legend()
    plt.tight_layout()
    plt.savefig('Test-Combined-new-new.png')


central, errs, nsq_order, jk_params = jackknife_fit(all_ratios, reg_low, reg_up, nconf, frozen=True)
print(central, errs, nsq_order, jk_params)
plot_all_t_with_bands(all_ratios, nconf, nsq_order, central, jk_params)
