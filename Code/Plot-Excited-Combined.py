import numpy as np
import matplotlib.pyplot as plt
import json
from FitModel import model_eq30  # deine Modellgleichung


import pandas as pd
import os
from pathlib import Path
from FitModel import model_eq30  # importiere deine Modellfunktion

# --- Configuration ---
ensemble = "C1"
cmass = '0.400'
FF = "A0"

fit_file = f"../Results/{ensemble}/{cmass}/Fit/Excited-combined-{FF}.csv"
out_plot = f"../Results/{ensemble}/{cmass}/Fit/Excited-combined-{FF}-Plot.pdf"

def load_ratios(ensemble, cmass, nsq_order):
    all_ratios = {}
    all_errs = {}
    for nsq in nsq_order:
        path_ratio = f"../Data/{ensemble}/{cmass}/ratios_{nsq}.npy"
        path_err   = f"../Data/{ensemble}/{cmass}/errors_{nsq}.npy"
        all_ratios[int(nsq)] = np.load(path_ratio, allow_pickle=True).item()
        all_errs[int(nsq)]   = np.load(path_err, allow_pickle=True).item()
    return all_ratios, all_errs

import numpy as np

fitparam_file = f"../Results/{ensemble}/{cmass}/Fit/Excited-combined-{FF}-fitparams.npz"
fit_data = np.load(fitparam_file, allow_pickle=True)

central = fit_data["central"]
errs = fit_data["errs"]
jk_params = fit_data["jk_params"]
nsq_order = fit_data["nsq_order"]
reg_low = int(fit_data["reg_low"])
reg_up = int(fit_data["reg_up"])
nconf = int(fit_data["nconf"])

# Jackknife-dependent inputs
mDs_gs_jk = fit_data["mDs_gs_jk"]
mDs_es_jk = fit_data["mDs_es_jk"]
mBs_gs_jk = fit_data["mBs_gs_jk"]
mBs_es_jk = fit_data["mBs_es_jk"]
Z0_Ds_jk  = fit_data["Z0_Ds_jk"]
Z1_Ds_jk  = fit_data["Z1_Ds_jk"]
Z0_Bs_jk  = fit_data["Z0_Bs_jk"]
Z1_Bs_jk  = fit_data["Z1_Bs_jk"]

print("✅ Loaded all fit and jackknife data successfully.")


# --- Read metadata + table ---
meta = {}
table_lines = []
with open(fit_file, "r") as f:
    for line in f:
        if line.startswith("#"):
            parts = line[1:].strip().split(":")
            if len(parts) == 2:
                meta[parts[0].strip()] = parts[1].strip()
        else:
            table_lines.append(line)

from io import StringIO
df = pd.read_csv(StringIO("".join(table_lines)), sep="\t")

# --- Extract values ---
reg_low = int(meta["reg_low"])
reg_up  = int(meta["reg_up"])
nsq_order = df["nsq"].astype(int).tolist()

O00, O01, O10 = df["O00"].to_numpy(), df["O01"].to_numpy(), df["O10"].to_numpy()
dO00, dO01, dO10 = df["dO00"].to_numpy(), df["dO01"].to_numpy(), df["dO10"].to_numpy()


meta_file = f"../Results/{ensemble}/{cmass}/Fit/Excited-combined-{FF}-meta.json"
with open(meta_file) as f:
    meta_all = json.load(f)

mDs_gs = {int(k): float(v) for k, v in meta_all["mDs_gs"].items()}
mDs_es = {int(k): float(v) for k, v in meta_all["mDs_es"].items()}
mBs_gs = float(meta_all["mBs_gs"])
mBs_es = float(meta_all["mBs_es"])
Z0_Ds = {int(k): float(v) for k, v in meta_all["Z0_Ds"].items()}
Z1_Ds = {int(k): float(v) for k, v in meta_all["Z1_Ds"].items()}
Z0_Bs = float(meta_all["Z0_Bs"])
Z1_Bs = float(meta_all["Z1_Bs"])
T = float(meta_all["T"])


def plot_all_t_eq30(
    all_ratios, all_errs, nconf, nsq_order, central, jk_params,
    mDs_gs, mDs_es, mBs_gs, mBs_es,
    Z0_Ds, Z1_Ds, Z0_Bs, Z1_Bs, T,
    mDs_gs_jk, mDs_es_jk, mBs_gs_jk, mBs_es_jk,
    Z0_Ds_jk, Z1_Ds_jk, Z0_Bs_jk, Z1_Bs_jk,
    reg_low, reg_up, outname="Ex-Fit.png"
):
    plt.figure(figsize=(8,6))
    colors = plt.cm.tab10.colors
    nsq_order = [int(x) for x in nsq_order]
    n = len(nsq_order)

    # Parameterblöcke
    O00_c = central[0*n:1*n]
    O01_c = central[1*n:2*n]
    O10_c = central[2*n:3*n]

    jk_params = np.array(jk_params)
    O00_jk = jk_params[:, 0*n:1*n]
    O01_jk = jk_params[:, 1*n:2*n]
    O10_jk = jk_params[:, 2*n:3*n]

    t_all = np.arange(1, 24)
    fit_mask = (t_all >= reg_low) & (t_all < reg_up)

    for i, nsq in enumerate(nsq_order):
        color = colors[i % len(colors)]
        data_mean = all_ratios[int(nsq)][:, -1]
        data_err  = all_errs[int(nsq)]

        # Datenpunkte
        t_data = np.arange(len(data_mean))
        mask = (t_data >= 1) & (t_data <= 23)
        plt.errorbar(t_data[mask], data_mean[mask], yerr=data_err[mask],
                     fmt='x', color=color, label=f'n²={nsq}')

        # Zentraler Fit
        mD0, mD1 = float(mDs_gs[int(nsq)]), float(mDs_es[int(nsq)])
        M_B0, M_B1 = float(mBs_gs), float(mBs_es)
        ZD0, ZD1 = float(Z0_Ds[int(nsq)]), float(Z1_Ds[int(nsq)])
        ZB0, ZB1 = float(Z0_Bs), float(Z1_Bs)
        first_nsq = nsq_order[0]
        M_D0 = float(mDs_gs[int(first_nsq)])
        M_D1 = float(mDs_es[int(first_nsq)])

        term00 = ZB0 * O00_c[i] * ZD0 * np.exp(-mD0*t_all - M_B0*(T - t_all)) / (4*mD0*M_B0)
        term10 = ZB1 * O10_c[i] * ZD0 * np.exp(-mD0*t_all - M_B1*(T - t_all)) / (4*mD0*M_B1)
        term01 = ZB0 * O01_c[i] * ZD1 * np.exp(-mD1*t_all - M_B0*(T - t_all)) / (4*mD1*M_B0)
        denom_Bs = (ZB0**2/(2*M_B0))*np.exp(-M_B0*(T-t_all)) + (ZB1**2/(2*M_B1))*np.exp(-M_B1*(T-t_all))
        denom_Ds = (ZD0**2/(2*M_D0))*np.exp(-mD0*t_all) + (ZD1**2/(2*M_D1))*np.exp(-mD1*t_all)
        pref = 4*mD0*M_B0 / np.exp(-mD0*t_all - M_B0*(T - t_all))
        fit_central = np.sqrt(pref)*(term00+term10+term01)/np.sqrt(denom_Bs*denom_Ds)

        plt.plot(t_all[fit_mask], fit_central[fit_mask], '-', color=color)

        # Jackknife-Band
        band = []
        for k in range(nconf):
            mD0_k = float(mDs_gs_jk[k][int(nsq)])
            mD1_k = float(mDs_es_jk[k][int(nsq)])
            MB0_k = float(mBs_gs_jk[k])
            MB1_k = float(mBs_es_jk[k])
            ZD0_k = float(Z0_Ds_jk[k][int(nsq)])
            ZD1_k = float(Z1_Ds_jk[k][int(nsq)])
            ZB0_k = float(Z0_Bs_jk[k])
            ZB1_k = float(Z1_Bs_jk[k])
            O00k, O01k, O10k = O00_jk[k,i], O01_jk[k,i], O10_jk[k,i]

            first_nsq = nsq_order[0]
            MD0_k = float(mDs_gs_jk[k][int(first_nsq)])
            MD1_k = float(mDs_es_jk[k][int(first_nsq)])

            term00 = ZB0_k * O00k * ZD0_k * np.exp(-mD0_k*t_all - MB0_k*(T - t_all)) / (4*mD0_k*MB0_k)
            term10 = ZB1_k * O10k * ZD0_k * np.exp(-mD0_k*t_all - MB1_k*(T - t_all)) / (4*mD0_k*MB1_k)
            term01 = ZB0_k * O01k * ZD1_k * np.exp(-mD1_k*t_all - MB0_k*(T - t_all)) / (4*mD1_k*MB0_k)
            denom_Bs = (ZB0_k**2/(2*MB0_k))*np.exp(-MB0_k*(T-t_all)) + (ZB1_k**2/(2*MB1_k))*np.exp(-MB1_k*(T-t_all))
            denom_Ds = (ZD0_k**2/(2*MD0_k))*np.exp(-mD0_k*t_all) + (ZD1_k**2/(2*MD1_k))*np.exp(-mD1_k*t_all)
            pref = 4*mD0_k*MB0_k / np.exp(-mD0_k*t_all - MB0_k*(T - t_all))
            band.append(np.sqrt(pref)*(term00+term10+term01)/np.sqrt(denom_Bs*denom_Ds))

        band = np.array(band)
        mean = band.mean(axis=0)
        err  = np.sqrt((nconf-1)*np.mean((band-mean)**2, axis=0))
        plt.fill_between(t_all[fit_mask], mean[fit_mask]-err[fit_mask],
                         mean[fit_mask]+err[fit_mask], color=color, alpha=0.3)

    plt.xlabel("t")
    plt.ylabel(r"$\tilde{A}_0(t)$")
    plt.ylim(0.09, 0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f'../Results/{ensemble}/{cmass}/Fit/'+outname, dpi=200)
    plt.show()


#all_ratios, all_errs = load_ratios(ensemble, cmass, nsq_order)

data = np.load(f"../Results/{ensemble}/{cmass}/Fit/Excited-combined-{FF}-data.npz", allow_pickle=True)
all_ratios = data["all_ratios"].item()
all_errs = data["all_errs"].item()

first_key = nsq_order[0]
print("DEBUG: keys in all_ratios:", list(all_ratios.keys()))
print(f"DEBUG: all_ratios[{first_key}] type:", type(all_ratios[first_key]))
print(f"DEBUG: all_ratios[{first_key}] shape:", getattr(all_ratios[first_key], "shape", "no shape"))

nconf = all_ratios[nsq_order[0]].shape[1] - 1 

plot_all_t_eq30(
    all_ratios, all_errs, nconf, nsq_order, central, jk_params,
    mDs_gs, mDs_es, mBs_gs, mBs_es,
    Z0_Ds, Z1_Ds, Z0_Bs, Z1_Bs, T,
    mDs_gs_jk, mDs_es_jk, mBs_gs_jk, mBs_es_jk,
    Z0_Ds_jk, Z1_Ds_jk, Z0_Bs_jk, Z1_Bs_jk,
    reg_low=reg_low, reg_up=reg_up,
    outname=f"Ex-Fit-{ensemble}-{cmass}-{FF}.png"
)
