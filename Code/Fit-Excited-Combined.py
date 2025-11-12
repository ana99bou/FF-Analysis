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
import os
 
#########Choose Params

'''
FF = sys.argv[1]
#nsq = int(sys.argv[2])
cmass_index = int(sys.argv[3])
ensemble = sys.argv[4]
use_disp=bool(int(sys.argv[5]))
frozen_analysis = bool(int(sys.argv[6]))
'''


FF='A1'
nsq=1
cmass_index=2
ensemble="F1S"
use_disp=True
frozen=False


#Ens.getCmass(ensemble) gives us an array of the different charm masses for each ens; chose which one
cmass=Ens.getCmass(ensemble)[cmass_index]

#Fit range
if ensemble == 'F1S':
    if FF == 'V':
        reg_up=19
        reg_low=13
    elif FF == 'A0':
        reg_up=19
        reg_low=8
    elif FF == 'A1':
        reg_up=19
        reg_low=12
elif ensemble in ['M1']:
    reg_up=17#18 fr M3 2 A0
    reg_low=7
elif ensemble in ['M3']:
    if FF == 'A1':
        reg_up=17
        reg_low=10
    else:
        reg_up=17#18 fr M3 2 A0
        reg_low=7
elif ensemble in ['M2']:
    reg_up=15
    reg_low=5
elif ensemble in ['C1']:
    reg_up=14
    reg_low=4
elif ensemble in ['C2']:
    if FF == 'V':
        reg_up=11
        reg_low=5
    elif FF == 'A0':
        reg_up=9
        reg_low=3
    if FF == 'A1':
        reg_up=11
        reg_low=5

reg_up=reg_up+1

# Get strange mass and smearing radius
sm=Ens.getSM(ensemble)
smass=Ens.getSmass(ensemble)


# Ensemble details
inv=Ens.getInvSpac(ensemble)
nconf,dt,ts,L= Ens.getEns(ensemble)
m,csw,zeta=Ens.getRHQparams(ensemble)

# Read in 3pt and 2pt data

f3pt = h5py.File("../Data/{}/BsDsStar_{}_3pt.h5".format(ensemble,ensemble), "r")
f2ptDs = h5py.File("../Data/{}/BsDsStar_{}_2ptDs.h5".format(ensemble,ensemble), "r")
f2ptBs = h5py.File("../Data/{}/BsDsStar_{}_2ptBs.h5".format(ensemble,ensemble), "r")

# Instead of reading nsq from sys.argv
# Define which nsq values you want to process:
if FF in ['V', 'A0']:
    nsq_values = [0, 1, 2, 3, 4, 5]  # adjust to your available data
elif FF in ['A1']:
    nsq_values = [0,1, 2, 4, 5]  # adjust to your available data

# Storage for results
all_ratios = {}
all_errs = {}
all_jackknife_blocks = {}
all_avn0 = {}

for nsq in nsq_values:
    print(f"Processing nsq = {nsq} ...")

    # --- This is basically your original code from 'if use_disp' onwards ---
    if use_disp:
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
        edlist = [
            pd.read_csv(f'../Data/{ensemble}/2pt/Ds{cmass}Result-{i}.csv', sep='\t', index_col=0).loc[0,'EffectiveMass']
            for i in range(6)
        ]
        ed = edlist[nsq]
        md = edlist[0]
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
    path = f'../Data/{ensemble}/2pt/Excited-comb-Blocks-Bs.csv'
    df = pd.read_csv(path)   # no sep/index_col needed now
    df.rename(columns={'m0': 'EffectiveMass'}, inplace=True)
    bsfit=df

    if use_disp:
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

    #os.makedirs(f"../Data/{ensemble}/{cmass}/", exist_ok=True)

    #np.save(f"../Data/{ensemble}/{cmass}/ratios_{nsq}.npy", all_ratios)
    #np.save(f"../Data/{ensemble}/{cmass}/errors_{nsq}.npy", all_errs)

outdir = f"../Results/{ensemble}/{cmass}/Fit/"
os.makedirs(outdir, exist_ok=True)
np.savez_compressed(
    os.path.join(outdir, f"Excited-combined-{FF}-data.npz"),
    all_ratios=all_ratios,
    all_errs=all_errs
)

print(f"✅ Saved ratio data to {outdir}Excited-combined-{FF}-data.npz")


###############################################################################
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

print(mDs_GS_by_nsq)
print(A0_new_by_nsq)

print(mDs_ES_by_nsq)
print(A1_new_by_nsq)

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

def model_eq30(params, tvals, nsq_order,
               mDs_gs, mDs_es,         # E_Ds^(0/1)(p)
               mBs_gs, mBs_es,         # M_Bs^(0/1)
               Z0_Ds, Z1_Ds,
               Z0_Bs, Z1_Bs,
               T):
    tvals = np.asarray(tvals, dtype=float)
    nsq_order = [int(x) for x in nsq_order]
    n = len(nsq_order)

    O00 = params[0*n:1*n]
    O01 = params[1*n:2*n]
    O10 = params[2*n:3*n]

    out = []

    # Rest masses (momentum = 0)
    # Use the lowest available momentum as reference (proxy for rest)
    first_nsq = sorted(mDs_gs.keys())[0]
    M_D0 = float(mDs_gs[first_nsq])
    M_D1 = float(mDs_es[first_nsq])


    for i, nsq in enumerate(nsq_order):
        E_D0 = float(mDs_gs[int(nsq)])   # E_Ds^(0)(p)
        E_D1 = float(mDs_es[int(nsq)])   # E_Ds^(1)(p)
        M_B0 = float(mBs_gs)
        M_B1 = float(mBs_es)

        ZD0 = float(Z0_Ds[int(nsq)])
        ZD1 = float(Z1_Ds[int(nsq)])
        ZB0 = float(Z0_Bs)
        ZB1 = float(Z1_Bs)

        t = tvals

        # numerator
        term00 = ZB0 * O00[i] * ZD0 * np.exp(-E_D0*t - M_B0*(T - t)) / (4.0 * E_D0 * M_B0)
        term10 = ZB1 * O10[i] * ZD0 * np.exp(-E_D0*t - M_B1*(T - t)) / (4.0 * E_D0 * M_B1)
        term01 = ZB0 * O01[i] * ZD1 * np.exp(-E_D1*t - M_B0*(T - t)) / (4.0 * E_D1 * M_B0)

        # denominators
        denom_Bs = ((ZB0**2)/(2.0*M_B0)) * np.exp(-M_B0*(T - t)) + ((ZB1**2)/(2.0*M_B1)) * np.exp(-M_B1*(T - t))
        denom_Ds = ((ZD0**2)/(2.0*M_D0)) * np.exp(-E_D0*t)       + ((ZD1**2)/(2.0*M_D1)) * np.exp(-E_D1*t)

        # prefactor
        pref = 4.0*E_D0*M_B0 / np.exp(-E_D0*t - M_B0*(T - t))

        R = np.sqrt(pref) * (term00 + term10 + term01) / np.sqrt(denom_Bs * denom_Ds)
        out.append(R)

    return np.concatenate(out)


def make_chi2_eq30(all_ratios, reg_low, reg_up, nsq_order, cov_inv,
                   E0_Ds_by_nsq, E1_Ds_by_nsq,
                   M0_Bs, M1_Bs,
                   Z0_Ds_by_nsq, Z1_Ds_by_nsq,
                   Z0_Bs, Z1_Bs,
                   T):
    def chi2_minuit(*params):
        t_fit = np.arange(reg_low, reg_up)
        data = np.concatenate([
            np.mean(all_ratios[int(nsq)][reg_low:reg_up, :], axis=1)
            #all_ratios[int(nsq)][reg_low:reg_up, nconf]
            for nsq in nsq_order
        ])
        model_vals = model_eq30(params, t_fit, nsq_order,
                                E0_Ds_by_nsq, E1_Ds_by_nsq, M0_Bs, M1_Bs,
                                Z0_Ds_by_nsq, Z1_Ds_by_nsq, Z0_Bs, Z1_Bs, T)
        diff = data - model_vals
        return float(diff @ cov_inv @ diff)
    return chi2_minuit

def model_eq30_with_errors(central, jk_params, tvals, nsq_order,
                           mDs_gs, mDs_es, mBs_gs, mBs_es,
                           Z0_Ds, Z1_Ds, Z0_Bs, Z1_Bs, T):
    
    nconf = len(jk_params)
    # Evaluate central curve
    mean_curve = model_eq30(central, tvals, nsq_order,
                            mDs_gs, mDs_es, mBs_gs, mBs_es,
                            Z0_Ds, Z1_Ds, Z0_Bs, Z1_Bs, T)

    # Evaluate each jackknife curve
    jk_curves = []
    for i in range(nconf):
        curve_i = model_eq30(jk_params[i], tvals, nsq_order,
                             mDs_gs, mDs_es, mBs_gs, mBs_es,
                             Z0_Ds, Z1_Ds, Z0_Bs, Z1_Bs, T)
        jk_curves.append(curve_i)
    jk_curves = np.array(jk_curves)

    # Jackknife mean and error
    jk_mean = np.mean(jk_curves, axis=0)
    jk_err  = np.sqrt((nconf - 1) * np.mean((jk_curves - jk_mean)**2, axis=0))
    return jk_mean, jk_err


def jackknife_fit(all_ratios, reg_low, reg_up, nconf,
                  ds_blocks_gs, ds_blocks_es, bs_blocks_gs, bs_blocks_es,
                  frozen=True, T=None):
    assert T is not None, "Pass the temporal extent T (=ts)."
    nsq_order = [int(x) for x in sorted(all_ratios.keys())]
    n = len(nsq_order)

    # build these once
    Z0_Ds_jk = []
    Z1_Ds_jk = []
    Z0_Bs_jk = []
    Z1_Bs_jk = []
    for i in range(nconf):
        #Z0_Ds_jk.append({nsq: float(np.mean(np.delete(Z0_Ds_blocks[nsq], i))) for nsq in nsq_order})
        #Z1_Ds_jk.append({nsq: float(np.mean(np.delete(Z1_Ds_blocks[nsq], i))) for nsq in nsq_order})
        Z0_Ds_jk.append({nsq: float(Z0_Ds_blocks[nsq][i]) for nsq in nsq_order})
        Z1_Ds_jk.append({nsq: float(Z1_Ds_blocks[nsq][i]) for nsq in nsq_order})
    #Z0_Bs_jk = [float(np.mean(np.delete(Z0_Bs_blocks, i))) for i in range(nconf)]
    #Z1_Bs_jk = [float(np.mean(np.delete(Z1_Bs_blocks, i))) for i in range(nconf)]
        Z0_Bs_jk.append(float(Z0_Bs_blocks[i]))
        Z1_Bs_jk.append(float(Z1_Bs_blocks[i]))


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
        #mDs_gs_jk.append({nsq: float(np.mean(np.delete(ds_blocks_gs[nsq], i))) for nsq in nsq_order})
        #mDs_es_jk.append({nsq: float(np.mean(np.delete(ds_blocks_es[nsq], i))) for nsq in nsq_order})
        #mBs_gs_jk.append(float(np.mean(np.delete(bs_blocks_gs, i))))
        #mBs_es_jk.append(float(np.mean(np.delete(bs_blocks_es, i))))
        mDs_gs_jk.append({nsq: float(ds_blocks_gs[nsq][i]) for nsq in nsq_order})
        mDs_es_jk.append({nsq: float(ds_blocks_es[nsq][i]) for nsq in nsq_order})
        mBs_gs_jk.append(float(bs_blocks_gs[i]))
        mBs_es_jk.append(float(bs_blocks_es[i]))        

    
    jk_samples = np.array([
        np.concatenate([all_ratios[int(nsq)][reg_low:reg_up, i] for nsq in nsq_order])
        for i in range(nconf)
    ])      # shape (N, ndata)

    if frozen:
        #(nconf - 1)
        cov =  (nconf - 1)*np.cov(jk_samples, rowvar=False, ddof=0)
        cov_inv = np.linalg.inv(cov)
    else:
        cov_inv = None

    print("cov shape:", cov.shape)
    print("cond(cov):", np.linalg.cond(cov))
    print("mean(diag)^0.5:", np.sqrt(np.mean(np.diag(cov))))

    # ---- initial guesses for Eq. (30) ----
    # Parameters per-nsq: [O00_i for all i] + [O01_i for all i] + [O10_i for all i]
    O00_0 = [float(np.mean(all_ratios[int(nsq)][reg_low:reg_up, -1])) for nsq in nsq_order]
    O01_0 = [0.5 for _ in nsq_order]
    O10_0 = [0.5 for _ in nsq_order]
    p0 = O00_0 + O01_0 + O10_0

    print("Initial guesses (Eq.30):")
    for i, nsq in enumerate(nsq_order):
        print(f"nsq={nsq}: O00={O00_0[i]:.4e}, O01={O01_0[i]:.4e}, O10={O10_0[i]:.4e}")


    # ---- central fit ----
    chi2_fun = make_chi2_eq30(
        all_ratios, reg_low, reg_up, nsq_order,
        cov_inv,
        mDs_gs_central, mDs_es_central,
        mBs_gs_central, mBs_es_central,
        Z0_Ds_by_nsq, Z1_Ds_by_nsq,
        Z0_Bs_central, Z1_Bs_central,
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
            cov_i = (nconf - 1)*np.cov(sub, rowvar=False, ddof=1)
            cov_inv_i = np.linalg.inv(cov_i)
        else:
            cov_inv_i = cov_inv

       
        chi2_fun_i = make_chi2_eq30(
            {int(nsq): all_ratios[int(nsq)][:, i][:, None] for nsq in nsq_order},
            reg_low, reg_up, nsq_order,
            cov_inv_i,
            mDs_gs_jk[i], mDs_es_jk[i],
            mBs_gs_jk[i], mBs_es_jk[i],
            Z0_Ds_jk[i], Z1_Ds_jk[i],
            Z0_Bs_jk[i], Z1_Bs_jk[i],
            T
        )

        mi = Minuit(chi2_fun_i, *p0)
        mi.migrad()
        jk_params.append(np.array(list(mi.values)))

    jk_params = np.array(jk_params)
    errs = np.sqrt((nconf - 1) * np.mean((jk_params - central) ** 2, axis=0))

    # ==========================================================
# === Save fit results for plotting ========================
# ==========================================================

    import os

# Create output folder
    outdir = f"../Results/{ensemble}/{cmass}/Fit/"
    os.makedirs(outdir, exist_ok=True)

    outpath = os.path.join(outdir, f"Excited-combined-{FF}.csv")

# Build a DataFrame
    n = len(nsq_order)
    df = pd.DataFrame({
        "nsq": nsq_order,
        "O00": central[0*n:1*n],
        "dO00": errs[0*n:1*n],
        "O01": central[1*n:2*n],
        "dO01": errs[1*n:2*n],
        "O10": central[2*n:3*n],
        "dO10": errs[2*n:3*n],
        })

    meta = {
        "ensemble": ensemble,
        "cmass": cmass,
        "FF": FF,
        "use_disp": use_disp,
        "frozen": frozen,
        "reg_low": reg_low,
        "reg_up": reg_up,
        "chi2": m.fval,
        "ndof": ndof,
        "chi2/dof": chi2/ndof,
        "pvalue": p_val,
    }

# Save metadata as header (human-readable)
    with open(outpath, "w") as f:
        f.write("# Excited combined fit results\n")
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        df.to_csv(f, sep="\t", index=False)

    print(f"Saved fit results to {outpath}")


    import json

    meta_all = {
    "ensemble": ensemble,
    "cmass": cmass,
    "FF": FF,
    "use_disp": use_disp,
    "frozen": frozen,
    "reg_low": reg_low,
    "reg_up": reg_up,
    "chi2": m.fval,
    "ndof": ndof,
    "chi2/dof": chi2/ndof,
    "pvalue": p_val,
    "T": float(dt),
    "mDs_gs": mDs_GS_by_nsq,
    "mDs_es": mDs_ES_by_nsq,
    "mBs_gs": float(mBs_GS_central),
    "mBs_es": float(mBs_ES_central),
    "Z0_Ds": {int(k): float(v) for k, v in Z0_Ds_by_nsq.items()},
    "Z1_Ds": {int(k): float(v) for k, v in Z1_Ds_by_nsq.items()},
    "Z0_Bs": float(Z0_Bs_central),
    "Z1_Bs": float(Z1_Bs_central),
    }

    json_path = os.path.join(outdir, f"Excited-combined-{FF}-meta.json")
    with open(json_path, "w") as f:
        json.dump(meta_all, f, indent=2)

    print(f"Saved meta info to {json_path}")


    #return central, errs, nsq_order, jk_params, (mDs_gs_central, mDs_es_central, mBs_gs_central, mBs_es_central)
    return (central, errs, nsq_order, jk_params,
            (mDs_gs_central, mDs_es_central, mBs_gs_central, mBs_es_central),
            mDs_gs_jk, mDs_es_jk, mBs_gs_jk, mBs_es_jk,
            Z0_Ds_jk, Z1_Ds_jk, Z0_Bs_jk, Z1_Bs_jk)

(
    central, errs, nsq_order, jk_params,
    masses_central,
    mDs_gs_jk, mDs_es_jk, mBs_gs_jk, mBs_es_jk,
    Z0_Ds_jk, Z1_Ds_jk, Z0_Bs_jk, Z1_Bs_jk
) = jackknife_fit(
    all_ratios, reg_low, reg_up, nconf,
    ds_blocks_gs, ds_blocks_es, bs_blocks_gs, bs_blocks_es,
    frozen=True, T=dt
)


print(central)
print(errs)

import os

outdir = f"../Results/{ensemble}/{cmass}/Fit/"
os.makedirs(outdir, exist_ok=True)

# Save fit results and jackknife inputs for later plotting
np.savez_compressed(
    os.path.join(outdir, f"Excited-combined-{FF}-fitparams.npz"),
    central=central,
    errs=errs,
    nsq_order=nsq_order,
    jk_params=jk_params,
    reg_low=reg_low,
    reg_up=reg_up,
    nconf=nconf,
    # jackknife-dependent inputs for uncertainty propagation
    mDs_gs_jk=mDs_gs_jk,
    mDs_es_jk=mDs_es_jk,
    mBs_gs_jk=mBs_gs_jk,
    mBs_es_jk=mBs_es_jk,
    Z0_Ds_jk=Z0_Ds_jk,
    Z1_Ds_jk=Z1_Ds_jk,
    Z0_Bs_jk=Z0_Bs_jk,
    Z1_Bs_jk=Z1_Bs_jk,
)

print(f"✅ Saved fit parameters to {outdir}Excited-combined-{FF}-fitparams.npz")

