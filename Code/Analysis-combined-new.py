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
from iminuit import Minuit
from scipy.stats import chi2 as chi2_dist
 
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
nsq=1
cmass_index=1
ensemble="F1S"
use_disp=True
frozen=True


#Ens.getCmass(ensemble) gives us an array of the different charm masses for each ens; chose which one
cmass=Ens.getCmass(ensemble)[cmass_index]

#Fit range
if ensemble == 'F1S':
    reg_up=25
    reg_low=7
elif ensemble in ['M1', 'M2', 'M3']:
    reg_up=15
    reg_low=7
elif ensemble in ['C1', 'C2']:
    reg_up=24
    reg_low=5

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
        #md = pd.read_csv(f'../Data/{ensemble}/2pt/Ds{cmass}Result-0.csv', sep='\t', index_col=0).loc[0, 'EffectiveMass']
        md = pd.read_csv(f'../Data/{ensemble}/2pt/Excited-comb-Ds{cmass}Result-0.csv', sep='\t', index_col=0).loc[0, 'Mass0']

        vecs = [[0,0,0],[1,0,0],[1,1,0],[1,1,1],[2,0,0],[2,1,0]]
        def calculate_value(a, md, p, L):
            term1 = np.sinh(md / 2) ** 2
            term2 = np.sin(p[0] * np.pi / L) ** 2
            term3 = np.sin(p[1] * np.pi / L) ** 2
            term4 = np.sin(p[2] * np.pi / L) ** 2
            return 2 * np.arcsinh(np.sqrt(term1 + term2 + term3 + term4))

        ed = calculate_value(1/inv, md, vecs[nsq], L)

    else:
        #not updated path to excited comb fit
        edlist = [
            pd.read_csv(f'../Data/{ensemble}/2pt/Ds{cmass}Result-{i}.csv', sep='\t', index_col=0).loc[0,'EffectiveMass']
            for i in range(6)
        ]
        ed = edlist[nsq]
        md = edlist[0]
    #mb=pd.read_csv('../Data/{}/2pt/BsResult.csv'.format(ensemble), sep='\t',index_col=0).loc[0,'EffectiveMass']
    mb=pd.read_csv('../Data/{}/2pt/Excited-comb-BsResult.csv'.format(ensemble), sep='\t',index_col=0).loc[0,'Mass0']

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
    #bsfit = pd.read_csv(f'../Data/{ensemble}/2pt/Bs-blocks.csv', sep='\s', engine="python")
    #bsfit = pd.read_csv('../Data/{}/2pt/Excited-comb-Bs-blocks.csv'.format(ensemble), sep='\t',index_col=0).loc['m0']
    path = f'../Data/{ensemble}/2pt/Excited-comb-Blocks-Bs.csv'
    df = pd.read_csv(path)   # no sep/index_col needed now
    df.rename(columns={'m0': 'EffectiveMass'}, inplace=True)
    #bsfit = df["EffectiveMass"].to_numpy().flatten()
    bsfit=df

    if use_disp:
        #dsfit_0 = pd.read_csv(f'../Data/{ensemble}/2pt/Ds{cmass}-nsq0-blocks.csv', sep='\s', engine="python")
        #dsfit_0 = pd.read_csv('../Data/{}/2pt/Excited-comb-Blocks-Ds{}-0.csv'.format(ensemble,cmass), sep='\t',index_col=0).loc['m0']
        #md_values = dsfit_0['m0']
        path = f'../Data/{ensemble}/2pt/Excited-comb-Blocks-Ds{cmass}-0.csv'
        df = pd.read_csv(path)   # no sep/index_col needed now
        md_values = df["m0"].to_numpy().flatten()
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



decay_const_by_nsq = {}
for nsq in nsq_values:
    path = f'../Data/{ensemble}/2pt/Excited-comb-Ds{cmass}Result-{int(nsq)}.csv'
    val = float(pd.read_csv(path, sep='\t', index_col=0).loc[0, 'DeltaM'])
    decay_const_by_nsq[int(nsq)] = val

lam_blocks = {}
for nsq in nsq_values:
    path = f'../Data/{ensemble}/2pt/Excited-comb-Blocks-Ds{cmass}-{int(nsq)}.csv'
    df = pd.read_csv(path)   # no sep/index_col needed now
    arr = df["dm"].to_numpy().flatten()
    lam_blocks[int(nsq)] = arr  # store per nsq

#print(lam_blocks)


# ---------- Central masses and amplitudes (ground/excited) ----------

mDs_GS_by_nsq = {}
mDs_ES_by_nsq = {}
A0_old_by_nsq = {}
A1_old_by_nsq = {}
A0_new_by_nsq = {}
A1_new_by_nsq = {}

for nsq in nsq_values:
    path = f'../Data/{ensemble}/2pt/Excited-comb-Ds{cmass}Result-{int(nsq)}.csv'
    df = pd.read_csv(path, sep='\t', index_col=0)

    mDs_GS_by_nsq[int(nsq)] = float(df.loc[0, 'Mass0'])
    mDs_ES_by_nsq[int(nsq)] = float(df.loc[0, 'Mass1'])

    # new amplitude parameters
    A0_old_by_nsq[int(nsq)] = float(df.loc[0, 'A0_old'])
    A1_old_by_nsq[int(nsq)] = float(df.loc[0, 'A1_old'])
    A0_new_by_nsq[int(nsq)] = float(df.loc[0, 'A0_new'])
    A1_new_by_nsq[int(nsq)] = float(df.loc[0, 'A1_new'])

# Bs: nsq = 0 only (same file for all nsq in the fit)
dfb = pd.read_csv(f'../Data/{ensemble}/2pt/Excited-comb-BsResult.csv',
                  sep='\t', index_col=0)
mBs_GS_central = float(dfb.loc[0, 'Mass0'])
mBs_ES_central = float(dfb.loc[0, 'Mass1'])

# ---------- Compute central Z-factors ----------

Z0_Ds_by_nsq = {}
Z1_Ds_by_nsq = {}

for nsq in nsq_values:
    m0 = mDs_GS_by_nsq[int(nsq)]
    m1 = mDs_ES_by_nsq[int(nsq)]
    A0_new = A0_new_by_nsq[int(nsq)]
    A1_new = A1_new_by_nsq[int(nsq)]
    Z0_Ds_by_nsq[int(nsq)] = np.sqrt(2 * m0 * A0_new)
    Z1_Ds_by_nsq[int(nsq)] = np.sqrt(2 * m1 * A1_new)

# Bs (no nsq dependence)
m0_Bs = mBs_GS_central
m1_Bs = mBs_ES_central
A0_new_Bs = float(dfb.loc[0, 'A0_new'])
A1_new_Bs = float(dfb.loc[0, 'A1_new'])

Z0_Bs_central = np.sqrt(2 * m0_Bs * A0_new_Bs)
Z1_Bs_central = np.sqrt(2 * m1_Bs * A1_new_Bs)

print("Central Z-factors computed:")
print("Ds:", {n: (Z0_Ds_by_nsq[n], Z1_Ds_by_nsq[n]) for n in nsq_values})
print("Bs:", Z0_Bs_central, Z1_Bs_central)



# ---------- Jackknife blocks for masses and amplitudes ----------

# Ds: per nsq, columns "m0" (GS), "m1" (ES), plus amplitude columns
ds_blocks_gs = {}
ds_blocks_es = {}
A0_old_blocks = {}
A1_old_blocks = {}
A0_new_blocks = {}
A1_new_blocks = {}

for nsq in nsq_values:
    df = pd.read_csv(f'../Data/{ensemble}/2pt/Excited-comb-Blocks-Ds{cmass}-{int(nsq)}.csv')

    ds_blocks_gs[int(nsq)] = df['m0'].to_numpy().astype(float)
    ds_blocks_es[int(nsq)] = df['m1'].to_numpy().astype(float)

    # new amplitude jackknife arrays
    A0_old_blocks[int(nsq)] = df['A0_old'].to_numpy().astype(float)
    A1_old_blocks[int(nsq)] = df['A1_old'].to_numpy().astype(float)
    A0_new_blocks[int(nsq)] = df['A0_new'].to_numpy().astype(float)
    A1_new_blocks[int(nsq)] = df['A1_new'].to_numpy().astype(float)

# Bs: single set (nsq=0), columns "m0" (GS), "m1" (ES)
dfb_blocks = pd.read_csv(f'../Data/{ensemble}/2pt/Excited-comb-Blocks-Bs.csv')
bs_blocks_gs = dfb_blocks['m0'].to_numpy().astype(float)
bs_blocks_es = dfb_blocks['m1'].to_numpy().astype(float)

# ---------- Jackknife Z-factors ----------

Z0_Ds_blocks = {}
Z1_Ds_blocks = {}

for nsq in nsq_values:
    m0_blocks = ds_blocks_gs[int(nsq)]
    m1_blocks = ds_blocks_es[int(nsq)]
    A0_blocks = A0_new_blocks[int(nsq)]
    A1_blocks = A1_new_blocks[int(nsq)]

    Z0_Ds_blocks[int(nsq)] = np.sqrt(2 * m0_blocks * A0_blocks)
    Z1_Ds_blocks[int(nsq)] = np.sqrt(2 * m1_blocks * A1_blocks)

# Bs (no nsq dependence)
m0_blocks_Bs = bs_blocks_gs
m1_blocks_Bs = bs_blocks_es
A0_blocks_Bs = dfb_blocks['A0_new'].to_numpy().astype(float)
A1_blocks_Bs = dfb_blocks['A1_new'].to_numpy().astype(float)

Z0_Bs_blocks = np.sqrt(2 * m0_blocks_Bs * A0_blocks_Bs)
Z1_Bs_blocks = np.sqrt(2 * m1_blocks_Bs * A1_blocks_Bs)

print("Z-blocks shapes:")
print("Ds Z0:", {n: Z0_Ds_blocks[n].shape for n in nsq_values})
print("Bs Z0:", Z0_Bs_blocks.shape)



def model(params, tvals, nsq_order, mDs_gs, mDs_es, mBs_gs, mBs_es, T):
    """
    params: [R0_i (i=0..n-1), R1_i (i=0..n-1), R2_i (i=0..n-1)]
    mDs_gs/es: dict nsq -> float masses
    mBs_gs/es: float masses (same for all nsq)
    T: temporal extent (use 'ts' from Ens.getEns)
    """
    n = len(nsq_order)
    R0 = params[:n]
    R1 = params[n:2*n]
    R2 = params[2*n:3*n]

    out = []
    for i, nsq in enumerate(nsq_order):
        mD_es = float(mDs_es[int(nsq)])
        mD_gs = float(mDs_gs[int(nsq)])
        # If your physics requires decaying exponentials, change the four '+' to '-' below.
        term = (
            R0[i]
            + R1[i] * np.exp(-mD_es * tvals) * np.exp(-mBs_gs * (dt - tvals))
            + R2[i] * np.exp(-mD_gs * tvals) * np.exp(-mBs_es * (dt - tvals))
        )
        out.append(term)
    return np.concatenate(out)


def make_chi2(all_ratios, reg_low, reg_up, nsq_order, cov_inv,
              mDs_gs, mDs_es, mBs_gs, mBs_es, T):
    def chi2_minuit(*params):
        t_fit = np.arange(reg_low, reg_up)
        data = np.concatenate([
            np.mean(all_ratios[int(nsq)][reg_low:reg_up, :], axis=1)
            for nsq in nsq_order
        ])
        diff = data - model(params, t_fit, nsq_order, mDs_gs, mDs_es, mBs_gs, mBs_es, T)
        return float(diff @ cov_inv @ diff)
    return chi2_minuit


def jackknife_fit(all_ratios, reg_low, reg_up, nconf,
                  ds_blocks_gs, ds_blocks_es, bs_blocks_gs, bs_blocks_es,
                  frozen=True, T=None):
    assert T is not None, "Pass the temporal extent T (=ts)."
    nsq_order = [int(x) for x in sorted(all_ratios.keys())]
    n = len(nsq_order)

    # ---- central masses (use directly from previous fits, do NOT average blocks) ----
    # these were already read from the Excited-comb-*.csv results before calling jackknife_fit
    global mDs_GS_by_nsq, mDs_ES_by_nsq, mBs_GS_central, mBs_ES_central

    mDs_gs_central = {nsq: mDs_GS_by_nsq[int(nsq)] for nsq in nsq_order}
    mDs_es_central = {nsq: mDs_ES_by_nsq[int(nsq)] for nsq in nsq_order}
    mBs_gs_central = mBs_GS_central
    mBs_es_central = mBs_ES_central


    # ---- jackknife masses (leave-one-out means) ----
    mDs_gs_jk = []
    mDs_es_jk = []
    mBs_gs_jk = []
    mBs_es_jk = []
    for i in range(nconf):
        mDs_gs_jk.append({nsq: float(np.mean(np.delete(ds_blocks_gs[nsq], i))) for nsq in nsq_order})
        mDs_es_jk.append({nsq: float(np.mean(np.delete(ds_blocks_es[nsq], i))) for nsq in nsq_order})
        mBs_gs_jk.append(float(np.mean(np.delete(bs_blocks_gs, i))))
        mBs_es_jk.append(float(np.mean(np.delete(bs_blocks_es, i))))

    # ---- data stacking for covariance (central/frozen) ----
    jk_samples = np.array([
        np.concatenate([
            all_ratios[int(nsq)][reg_low:reg_up, i]
            for nsq in nsq_order
        ])
        for i in range(nconf)
    ])
    if frozen:
        cov = np.cov(jk_samples, rowvar=False, ddof=1)
        cov_inv = np.linalg.inv(cov)
    else:
        cov_inv = None  # will rebuild inside loop

    # ---- initial guess: R0 ~ time-avg, R1,R2 ~ 0 ----
    # ---- initial guesses based on 2pt function values and exponentials ----
    R0_0 = []
    R1_0 = []
    R2_0 = []

    for nsq in nsq_order:
        ratio_central = all_ratios[int(nsq)][:, -1]  # central ratio (mean over configs)
        R0_guess = float(np.mean(ratio_central))     # mean over all time slices
        R0_0.append(R0_guess)

    # get masses for this nsq
        mD_gs = float(mDs_gs_central[int(nsq)])
        mD_es = float(mDs_es_central[int(nsq)])
        mB_gs = float(mBs_gs_central)
        mB_es = float(mBs_es_central)

    # 2pt correlation functions for Ds and Bs (use averaged/folded central values)
    # here we can reuse your avdx and avb arrays if accessible, but since we are
    # in jackknife_fit, we reconstruct a reasonable proxy from all_ratios.
    # If you have the real 2pt central correlators stored externally, substitute them here.
    #
    # For now we assume C_2pt ~ R0_guess scale, adjust if you have proper 2pt values.
        C_reg_low = float(all_avn0[int(nsq)][reg_low])
        C_reg_up  = float(all_avn0[int(nsq)][reg_up]) - 1  # -1 because reg_up is exclusive in Python slicing

    # exponential factors
        exp1 = np.exp(mD_es * reg_low) * np.exp(mB_gs * (dt - reg_low))
        exp2 = np.exp(mD_gs * reg_up)  * np.exp(mB_es * (dt - reg_up))

    # initial guesses for R1 and R2
        R1_guess = (C_reg_low-R0_guess) * exp1
        R2_guess = (C_reg_up-R0_guess)  * exp2

        #R1_guess=0.1
        #R2_guess=0.1

        R1_0.append(R1_guess)
        R2_0.append(R2_guess)

        # flatten all parameters into one vector
    p0 = R0_0 + R1_0 + R2_0

    print("Initial guesses:")
    for i, nsq in enumerate(nsq_order):
        print(f"nsq={nsq}: R0={R0_0[i]:.4e}, R1={R1_0[i]:.4e}, R2={R2_0[i]:.4e}")


    # ---- central fit ----
    chi2_fun = make_chi2(
        all_ratios, reg_low, reg_up, nsq_order,
        cov_inv,
        mDs_gs_central, mDs_es_central,
        mBs_gs_central, mBs_es_central,
        T
    )
    m = Minuit(chi2_fun, *p0)
    m.migrad()

    ndata = len(np.arange(reg_low, reg_up)) * len(nsq_order)
    ndof  = ndata - m.nfit
    chi2  = m.fval
    p_val = 1 - chi2_dist.cdf(chi2, df=ndof)
    print('chi^2', chi2, 'ndof', ndof, 'chi^2/dof', chi2/ndof, 'p', p_val)

    central = np.array(list(m.values))

    # ---- jackknife fits ----
    jk_params = []
    for i in range(nconf):
        if not frozen:
            sub = np.delete(jk_samples, i, axis=0)
            cov_i = np.cov(sub, rowvar=False, ddof=1)
            cov_inv_i = np.linalg.inv(cov_i)
        else:
            cov_inv_i = cov_inv

        chi2_fun_i = make_chi2(
            {int(nsq): np.delete(all_ratios[int(nsq)], i, axis=1) for nsq in nsq_order},
            reg_low, reg_up, nsq_order,
            cov_inv_i,
            mDs_gs_jk[i], mDs_es_jk[i],
            mBs_gs_jk[i], mBs_es_jk[i],
            T
        )
        mi = Minuit(chi2_fun_i, *p0)
        mi.migrad()
        jk_params.append(np.array(list(mi.values)))

    jk_params = np.array(jk_params)
    errs = np.sqrt((nconf - 1) * np.mean((jk_params - central) ** 2, axis=0))
    return central, errs, nsq_order, jk_params, (mDs_gs_central, mDs_es_central, mBs_gs_central, mBs_es_central)

central, errs, nsq_order, jk_params, masses_central = jackknife_fit(
    all_ratios, reg_low, reg_up, nconf,
    ds_blocks_gs, ds_blocks_es, bs_blocks_gs, bs_blocks_es,
    frozen=True, T=ts   # <- 'ts' you already have from Ens.getEns(...)
)

print(central)
print(errs)

def plot_all_t_with_bands(all_ratios, nconf, nsq_order, central, jk_params,
                          mDs_gs, mDs_es, mBs_gs, mBs_es, T):
    plt.figure(figsize=(8,6))
    colors = plt.cm.tab10.colors
    nsq_order = [int(x) for x in nsq_order]
    n = len(nsq_order)

    R0_c = central[:n]
    R1_c = central[n:2*n]
    R2_c = central[2*n:3*n]

    jk_params = np.array(jk_params)
    R0_jk = jk_params[:, :n]
    R1_jk = jk_params[:, n:2*n]
    R2_jk = jk_params[:, 2*n:3*n]

    for i, nsq in enumerate(nsq_order):
        nt = all_ratios[int(nsq)].shape[0]
        t_all = np.arange(nt)
        data_mean = all_ratios[int(nsq)][:, -1]
        data_err  = np.std(all_ratios[int(nsq)][:, :nconf], axis=1, ddof=1)/np.sqrt(nconf)

        plt.errorbar(t_all, data_mean, yerr=data_err, fmt='x',
                     color=colors[i % len(colors)], label=f'nsq={nsq}')

        # central curve
        mD_es = float(mDs_es[int(nsq)])
        mD_gs = float(mDs_gs[int(nsq)])
        fit_c = (R0_c[i]
                 + R1_c[i]*np.exp(-mD_es*t_all)*np.exp(-mBs_gs*(dt - t_all))
                 + R2_c[i]*np.exp(-mD_gs*t_all)*np.exp(-mBs_es*(dt - t_all)))
        plt.plot(t_all, fit_c, '-', color=colors[i % len(colors)])

        # jackknife band
        band = []
        for k in range(nconf):
            R0k, R1k, R2k = R0_jk[k, i], R1_jk[k, i], R2_jk[k, i]
            band.append(R0k + R1k*np.exp(-mD_es*t_all)*np.exp(-mBs_gs*(dt - t_all))
                            + R2k*np.exp(-mD_gs*t_all)*np.exp(-mBs_es*(dt - t_all)))
        band = np.array(band)
        mean = np.mean(band, axis=0)
        err  = np.sqrt((nconf - 1) * np.mean((band - mean)**2, axis=0))
        plt.fill_between(t_all, mean - err, mean + err,
                         color=colors[i % len(colors)], alpha=0.3)

    plt.xlabel("t"); plt.ylabel("Ratio")
    plt.xlim(0, 24)
    plt.ylim(0, 1.5)
    plt.title("Global fit with GS/ES(Ds,Bs) terms")
    plt.legend(); plt.tight_layout()
    plt.savefig('Fit-GSES-model.png')

mDs_gs_c, mDs_es_c, mBs_gs_c, mBs_es_c = masses_central
#plot_all_t_with_bands(all_ratios, nconf, nsq_order, central, jk_params,
                      #mDs_gs_c, mDs_es_c, mBs_gs_c, mBs_es_c, ts)




'''
def model(params, tvals, nsq_order, lam_by_nsq):
    nsq_order = [int(x) for x in nsq_order]
    n = len(nsq_order)
    A = params[:n]
    B = params[n:2*n]

    out = []
    for i, nsq in enumerate(nsq_order):
        lam = float(lam_by_nsq[int(nsq)])  # ensure pure float
        out.append(A[i] + B[i] * np.exp(-lam * tvals))
    return np.concatenate(out)

def chi2(params, all_ratios, reg_low, reg_up, nsq_order, cov_inv, lam_by_nsq):
    t_fit = np.arange(reg_low, reg_up)
    # stack data in the same order as model’s concatenation
    #data = np.concatenate([
    #    np.mean(all_ratios[int(nsq)][reg_low:reg_up, :], axis=1)
    #    for nsq in nsq_order
    #])
    data = np.concatenate([
    all_ratios[int(nsq)][reg_low:reg_up, -1]   # last column = central value
    for nsq in nsq_order
    ])
    diff = data - model(params, t_fit, nsq_order, lam_by_nsq)
    return diff @ cov_inv @ diff

def make_chi2(all_ratios, reg_low, reg_up, nsq_order, cov_inv, lam_by_nsq):
    def chi2_minuit(*params):
        t_fit = np.arange(reg_low, reg_up)
        data = np.concatenate([
            np.mean(all_ratios[int(nsq)][reg_low:reg_up, :], axis=1)
            for nsq in nsq_order
        ])
        diff = data - model(params, t_fit, nsq_order, lam_by_nsq)
        return float(diff @ cov_inv @ diff)
    return chi2_minuit




def jackknife_fit(all_ratios, reg_low, reg_up, nconf, lam_blocks, frozen=True):
    nsq_order = sorted(all_ratios.keys())
    nsq_order = [int(x) for x in nsq_order]
    n = len(nsq_order)

    # --- central λ dict from all configs ---
    lam_central = {nsq: np.mean(lam_blocks[nsq]) for nsq in nsq_order}

    # --- jackknife λ dicts ---
    lam_jk_dicts = []
    for i in range(nconf):
        lam_jk_dicts.append({nsq: np.mean(np.delete(lam_blocks[nsq], i)) for nsq in nsq_order})

    # --- build jackknife samples of ratios (like before) ---
    jk_samples = np.array([
    np.concatenate([
        all_ratios[int(nsq)][reg_low:reg_up, i]    # <-- TAKE COLUMN i, don't average
        for nsq in nsq_order
    ])
    for i in range(nconf)
    ])

    #jk_samples = []
    #for i in range(nconf):
    #    jk_samples.append(np.concatenate([
    #        np.mean(np.delete(all_ratios[int(nsq)], i, axis=1)[reg_low:reg_up, :], axis=1)
    #        for nsq in nsq_order
    #    ]))
    jk_samples = np.array(jk_samples)


    cov_inv=covmat_all


    # initial guess
    A0 = [np.mean(np.mean(all_ratios[int(nsq)][reg_low:reg_up, :], axis=1)) for nsq in nsq_order]
    print(lam_central[int(nsq)])
    print(all_ratios[int(nsq)][reg_low])
    B0 = [(np.mean(all_ratios[int(nsq)][reg_low])-np.mean(np.mean(all_ratios[int(nsq)][reg_low:reg_up, :], axis=1)))*np.exp(float(lam_central[int(nsq)])*reg_low) for nsq in nsq_order]
    p0 = A0 + B0

    print("Initial guess:", p0)

    # --- central fit ---
    chi2_fun = make_chi2(all_ratios, reg_low, reg_up, nsq_order, cov_inv, lam_central)
    m = Minuit(chi2_fun, *p0)  # only fit parameters go here
    m.tol = 1e-16/0.002 
    
    for i in range(5, 10):
        m.limits[i] = (None, 0.0) 
    
    m.migrad()

    #ndof = (reg_up - reg_low + 1) - m.nfit
    ndata = len(np.arange(reg_low, reg_up)) * len(nsq_order)
    ndof  = ndata - m.nfit

    chi2 = m.fval
    chi2_dof = chi2 / ndof
    p_value = 1 - chi2_dist.cdf(chi2, df=ndof)
    print('chi^2',chi2,"p-value", p_value)       

    print(chi2, ndof, chi2_dof)

    central = np.array(list(m.values))
    p0_central = central.tolist()

    print("Central fit params:", central)




    # --- jackknife fits ---
    jk_params = []
    for i in range(nconf):
        if not frozen:
            sub_jk = []
            for j in range(nconf):
                if j == i:
                    continue
                sub_jk.append(np.concatenate([
                    np.mean(np.delete(np.delete(all_ratios[int(nsq)], i, axis=1), j, axis=1)[reg_low:reg_up, :], axis=1)
                    for nsq in nsq_order
                ]))
            cov_i = np.cov(np.array(sub_jk), rowvar=False, ddof=1)
            cov_inv_i = np.linalg.inv(cov_i)
        else:
            cov_inv_i = cov_inv

        chi2_fun_i = make_chi2(
            {int(nsq): np.delete(all_ratios[int(nsq)], i, axis=1) for nsq in nsq_order},
            reg_low, reg_up, nsq_order, cov_inv_i, lam_jk_dicts[i]
        )
        m = Minuit(chi2_fun_i, *p0)
        #m = Minuit(chi2_fun_i, *p0_central)
        m.migrad()
        jk_params.append(np.array(list(m.values)))



    #jk_params = np.array(jk_params)
    jk_params = np.array(jk_params)          # convert list -> array
    jk_mean   = jk_params.mean(axis=0)
    #errs = np.sqrt((nconf - 1) * np.mean((jk_params - jk_mean) ** 2, axis=0))   
    errs = np.sqrt((nconf - 1) * np.mean((jk_params - central) ** 2, axis=0))   

    #errs = np.sqrt(((nconf - 1) / nconf) * np.mean((jk_params - central) ** 2, axis=0))
    return central, errs, nsq_order, jk_params, lam_central

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
        fit_curve = A_vals[i] + B_vals[i] * np.exp(-decay_consts[nsq] * t_all)
        plt.plot(t_all, fit_curve, '-', color=colors[i % len(colors)])

    plt.xlabel("t")
    plt.ylabel("Ratio")
    plt.title(f"Global fit: A_i + B_i * exp(-{decay_consts} * t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('Test-Combined-new.png')

def plot_all_t_with_bands(all_ratios, nconf, nsq_order, central, jk_params, lam_by_nsq):
    plt.figure(figsize=(8,6))
    colors = plt.cm.tab10.colors
    nsq_order = [int(x) for x in nsq_order]
    n = len(nsq_order)

    A_c = central[:n]
    B_c = central[n:2*n]

    jk_params = np.array(jk_params)
    A_jk = jk_params[:, :n]
    B_jk = jk_params[:, n:2*n]

    for i, nsq in enumerate(nsq_order):
        nt = all_ratios[int(nsq)].shape[0]
        t_all = np.arange(nt)
        data_mean = all_avn0[int(nsq)]
        data_err = all_errs[int(nsq)]

        plt.errorbar(t_all, data_mean, yerr=data_err, fmt='x',
                     color=colors[i % len(colors)], label=f'nsq={nsq}')

        lam = float(lam_by_nsq[int(nsq)])
        fit_central = A_c[i] + B_c[i] * np.exp(-lam * t_all)

        fit_jk_vals = []
        for k in range(nconf):
            A_k = A_jk[k, i]
            B_k = B_jk[k, i]
            fit_jk_vals.append(A_k + B_k * np.exp(-lam * t_all))
        fit_jk_vals = np.array(fit_jk_vals)

        fit_mean = np.mean(fit_jk_vals, axis=0)
        fit_err = np.sqrt((nconf - 1) * np.mean((fit_jk_vals - fit_mean)**2, axis=0))

        plt.plot(t_all, fit_central, '-', color=colors[i % len(colors)])
        plt.fill_between(t_all, fit_mean - fit_err, fit_mean + fit_err,
                         color=colors[i % len(colors)], alpha=0.3)

    plt.xlabel("t")
    plt.ylabel("Ratio")
    plt.title("Global fit with per-nsq decay constants")
    plt.legend()
    plt.tight_layout()
    plt.savefig('Test-Combined-with-per-nsq-decays.png')




central, errs, nsq_order, jk_params, lam_central = jackknife_fit(
    all_ratios, reg_low, reg_up, nconf, lam_blocks=lam_blocks, frozen=True
)
print(central)
print(errs)
#plot_all_t_with_bands(all_ratios, nconf, nsq_order, central, jk_params, lam_by_nsq=decay_const_by_nsq)
plot_all_t_with_bands(all_ratios, nconf, nsq_order, central, jk_params, lam_by_nsq=lam_central)

# Evaluate chi^2 at the central fit
# Rebuild cov_inv exactly as in jackknife_fit for central chi2
nsq_order_sorted = sorted(all_ratios.keys())
jk_samples = []
for i in range(nconf):
    jk_samples.append(np.concatenate([
        np.mean(np.delete(all_ratios[int(nsq)], i, axis=1)[reg_low:reg_up, :], axis=1)
        for nsq in nsq_order_sorted
    ]))
jk_samples = np.array(jk_samples)
cov = np.cov(jk_samples, rowvar=False, ddof=1)
cov_inv = np.linalg.inv(cov)



'''