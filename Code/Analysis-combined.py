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

def model_allnsq(params, t_slices, nsq_order):
    """
    Build the model vector for all nsq, concatenated.

    params: [A_nsq1, A_nsq2, ..., A_nsqN, m_d]
    t_slices: dict {nsq: np.array of time slices used in fit}
    nsq_order: list of nsq in the order used in the covariance building
    """
    n_nsq = len(nsq_order)
    A_vals = params[:n_nsq]
    m_d = params[-1]

    model_parts = []
    for i, nsq in enumerate(nsq_order):
        tvals = t_slices[nsq]
        model_parts.append(A_vals[i]+ np.exp(-(0.7491479301452659+m_d) * tvals))
    
    return np.concatenate(model_parts)

def chi2_allnsq(params, data_vec, invcov, t_slices, nsq_order):
    model_vec = model_allnsq(params, t_slices, nsq_order)
    diff = data_vec - model_vec
    return diff @ invcov @ diff

# Build the data vector in the same stacking order as the covariance
t_slices = {}
data_blocks = []
for nsq in nsq_order:
    tvals = np.arange(reg_low, reg_up)
    t_slices[nsq] = tvals
    block_mean = np.mean(all_ratios[nsq][reg_low:reg_up, :nconf], axis=1)
    data_blocks.append(block_mean)
data_vec = np.concatenate(data_blocks)

# Inverse covariance
invcov_all = np.linalg.inv(covmat_all)

# Initial guess: A for each nsq, plus m_d
p0 = [data_blocks[i][0] for i in range(len(nsq_order))] + [0.5]

# Minimize chi²
res = minimize(
    chi2_allnsq,
    x0=p0,
    args=(data_vec, invcov_all, t_slices, nsq_order),
    method='Nelder-Mead'
)

print("Fit result:", res.x)
print("Final chi²:", chi2_allnsq(res.x, data_vec, invcov_all, t_slices, nsq_order))

def jackknife_fit_allnsq(all_ratios, reg_low, reg_up, nconf, frozen=True):
    nsq_order = sorted(all_ratios.keys())
    t_slices = {nsq: np.arange(reg_low, reg_up) for nsq in nsq_order}

    fit_results = []

    # Frozen covariance (if desired)
    if frozen:
        covmat_all, _ = build_Covarianz_allnsq(all_ratios, reg_up, reg_low, nconf)
        invcov_all = np.linalg.inv(covmat_all)

    # Loop over jackknife samples
    for jk in range(nconf):
        # Build data vector from this jackknife
        data_blocks = []
        for nsq in nsq_order:
            block = np.delete(all_ratios[nsq][reg_low:reg_up, :], jk, axis=1)
            block_mean = np.mean(block, axis=1)
            data_blocks.append(block_mean)
        data_vec = np.concatenate(data_blocks)

        # Unfrozen covariance
        if not frozen:
            covmat_all, _ = build_Covarianz_allnsq(
                {nsq: np.delete(all_ratios[nsq], jk, axis=1) for nsq in nsq_order},
                reg_up, reg_low, nconf-1
            )
            invcov_all = np.linalg.inv(covmat_all)

        # Initial guess
        p0 = [data_blocks[i][0] for i in range(len(nsq_order))] + [0.5]

        # Minimize
        res = minimize(
            chi2_allnsq,
            x0=p0,
            args=(data_vec, invcov_all, t_slices, nsq_order),
            method='Nelder-Mead'
        )
        fit_results.append(res.x)

    # Convert to array for error analysis
    fit_results = np.array(fit_results)
    central = np.mean(fit_results, axis=0)
    errs = np.sqrt((nconf - 1) * np.mean((fit_results - central) ** 2, axis=0))

    return central, errs, nsq_order


import matplotlib.pyplot as plt
import numpy as np

def plot_allnsq_fit_all_t(all_ratios, reg_low, reg_up, nconf, nsq_order, central):
    plt.figure(figsize=(8,6))
    colors = plt.cm.tab10.colors  # distinct colors

    n_nsq = len(nsq_order)
    A_vals = central[:n_nsq]
    m_d = central[-1]

    for i, nsq in enumerate(nsq_order):
        # All time slices
        nt = all_ratios[nsq].shape[0]
        t_all = np.arange(nt)

        # Data mean and errors over all t
        data_mean = np.mean(all_ratios[nsq][:, :nconf], axis=1)
        data_err = np.std(all_ratios[nsq][:, :nconf], axis=1, ddof=1) / np.sqrt(nconf)

        # Plot all data points
        plt.errorbar(
            t_all, data_mean, yerr=data_err,
            fmt='x', color=colors[i % len(colors)], label=f'nsq={nsq}'
        )

        # Fit curve over full t range
        fit_curve = A_vals[i] + np.exp(-(0.7491479301452659+m_d) * t_all)
        plt.plot(t_all, fit_curve, '-', color=colors[i % len(colors)])

    plt.xlabel("t")
    plt.ylabel("Ratio")
    plt.title("Global fit across all nsq (full t-range shown)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('Test-Combined.png')

central, errs, nsq_order = jackknife_fit_allnsq(all_ratios, reg_low, reg_up, nconf, frozen=True)
plot_allnsq_fit_all_t(all_ratios, reg_low, reg_up, nconf, nsq_order, central)









'''

if FF == 'A2':
    covmat=Regression.build_Covarianz_A2(reg_up,reg_low,ts,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,md,mb,ed,pre,dsfit,bsfit,L,A0fit,A1fit,avn0)
    invcovmat=np.linalg.inv(covmat)
    cut=ts/2-1-reg_up
    def chi(a):
        return (nconf)/(nconf-1)*np.dot(np.transpose([i-a for i in avn0[reg_low:reg_up]]),np.matmul(invcovmat,[i-a for i in avn0[reg_low:reg_up]]))
    mbar=minimize(chi,0.1,method='Nelder-Mead', tol=1e-8).x[0]

    def jackmass(t1,i):
        return (((jb3pt[t1,i]))/(np.sqrt(jbdx[t1,i]*jbb[dt-(t1),i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*(t1))*np.exp(-bsfit['EffectiveMass'][i]*(dt-(t1)))))*pre
    def jackmass(t1,i):
        return (((jb3pt[t1,i]))/(np.sqrt(jbdx[t1,i]*jbb[dt-(t1),i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*(t1))*np.exp(-bsfit['EffectiveMass'][i]*(dt-(t1)))))*pre
    def chijack(a,k):
        return np.dot(np.transpose([Regression.jackratio_A2(k,i + reg_low,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit) - a for i in range(int(ts / 2 - 1 - reg_low - cut))]),
                  np.matmul(invcovmat,
                            [Regression.jackratio_A2(k,i + reg_low,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit) - a for i in range(int(ts / 2 - 1 - reg_low - cut))]))
    
    #Std Deviatson for all jakcknife blocks
    jblocks=np.zeros(nconf)
    h=0
    for i in range(nconf):
        tmp=minimize(chijack,0.1,args=(i),method='Nelder-Mead', tol=1e-8).x[0]
        jblocks[i]=tmp
        h=h+(tmp-mbar)**2
    sigma=np.sqrt((nconf-1)/(nconf)*h)

else:
    covmat=Regression.build_Covarianz(reg_up, reg_low, ratiojack,nconf)
    
    #define fit function to be constant
    def fit_function(a):
        return a
    

    #mbar,sigma,pval=Regression.get_fit(reg_up,reg_low,covmat,nconf,ratiojack,fit_function)

    frozen_analysis = True  # Set True to use frozen cov matrix (default)

    if not frozen_analysis:
        mbar, sigma, pval = Regression.get_fit(
            reg_up, reg_low, covmat, nconf, ratiojack, fit_function, unfrozen=True
        )
    else:
        mbar, sigma, pval = Regression.get_fit(
            reg_up, reg_low, covmat, nconf, ratiojack, fit_function
        )



#np.savetxt('../Results/{}/Fits/{}/{}-nsq{}.txt'.format(ensemble,FF,FF,nsq), np.c_[np.absolute(avn0), errn0])
#np.save('../Results/{}/Fits/{}/{}-nsq{}.npy'.format(ensemble,FF,FF,nsq), jblocks)

#df4 = pd.DataFrame(columns=['EffectiveMass'])
#df4['EffectiveMass']=jblocks
#df4.to_csv('../Results/{}/Fits/{}/{}-nsq{}-Block.csv'.format(ensemble,FF,FF,nsq), sep='\t')

print(mbar,sigma)

f = plt.figure(figsize=(10,3))
ax = f.add_subplot(121)

ax.set_xlabel('time')
ax.set_ylabel(rf'$\widetilde{{{FF}}}$')
ax.errorbar(list(range(dt))[1:dt], avn0[1:dt], yerr=errn0[1:dt],fmt='x', label='nsq={}'.format(nsq))
plt.axhline(y = mbar, color = 'r', linestyle = 'dashed')
plt.fill_between(list(range(dt))[reg_low:reg_up], mbar+sigma, mbar-sigma, color='r',alpha=0.2)


df3 = pd.DataFrame([{
    'EffectiveMass': mbar,
    'Error': sigma,
    'RegUp': reg_up,
    'RegLow': reg_low
}])


df4 = pd.DataFrame(columns=['pval'])
if FF == 'A2':
    #df4['pval']=Regression.pvalue(mbar.fun,reg_up-reg_low)
    pval=Regression.pvalue(mbar.fun,reg_up-reg_low)
    df4 = pd.DataFrame({'pval': [pval]})
else:
    #df4['pval']=pval
    df4 = pd.DataFrame({'pval': [pval]})


if use_disp:
    plt.savefig('../Results/{}/{}/Fits/{}/{}-Av-nsq{}-Fit-Disp.png'.format(ensemble,cmass,FF,FF,nsq))
    df3.to_csv('../Results/{}/{}/Fits/{}/{}-Av-nsq{}-Fit-Disp.csv'.format(ensemble,cmass,FF,FF,nsq), sep='\t')
    df4.to_csv('../Results/{}/{}/Fits/{}/pval-{}-nsq{}-Disp.csv'.format(ensemble,cmass,FF,FF,nsq), sep='\t')
else:
    plt.savefig('../Results/{}/{}/Fits/{}/{}-Av-nsq{}-Fit.png'.format(ensemble,cmass,FF,FF,nsq))
    df3.to_csv('../Results/{}/{}/Fits/{}/{}-Av-nsq{}-Fit.csv'.format(ensemble,cmass,FF,FF,nsq), sep='\t')
    df4.to_csv('../Results/{}/{}/Fits/{}/pval-{}-nsq{}.csv'.format(ensemble,cmass,FF,FF,nsq), sep='\t')
'''
