import glob
import re
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

############################################################
#  NEW: Ds masses in lattice units (val1 and valx)
############################################################
Ds_mass_phys = {
    "F1S": [
        0.756409336,  # nsq = 0
        0.766627017,  # nsq = 1
        0.77668534,   # nsq = 2
        0.786590562,  # nsq = 3
        0.796182786,  # nsq = 4
        0.805801389   # nsq = 5
    ],
    "M1": [
        0.883900474,  # nsq = 0
        0.902793628,  # nsq = 1
        0.921202063,  # nsq = 2
        0.939155484,  # nsq = 3
        0.956014792,  # nsq = 4
        0.973151066   # nsq = 5
    ],
    "M2": [
        0.883900474,  # nsq = 0
        0.902793628,  # nsq = 1
        0.921202063,  # nsq = 2
        0.939155484,  # nsq = 3
        0.956014792,  # nsq = 4
        0.973151066   # nsq = 5
    ],
    "M3": [
        0.883900474,  # nsq = 0
        0.902793628,  # nsq = 1
        0.921202063,  # nsq = 2
        0.939155484,  # nsq = 3
        0.956014792,  # nsq = 4
        0.973151066   # nsq = 5
    ],
    "C1": [
        1.180300314,  # nsq = 0
        1.203099766,  # nsq = 1
        1.225292858,  # nsq = 2
        1.246915416,  # nsq = 3
        1.266579507,  # nsq = 4
        1.287189271   # nsq = 5
    ],
    "C2": [
        1.180300314,  # nsq = 0
        1.203099766,  # nsq = 1
        1.225292858,  # nsq = 2
        1.246915416,  # nsq = 3
        1.266579507,  # nsq = 4
        1.287189271   # nsq = 5
    ],
}


############################################################
# Your renormalization-factor code (unchanged)
############################################################
variables = {}
for filename in glob.glob("../Data/Renorm/output-*.txt"):
    with open(filename, "r") as f:
        lines = f.readlines()
        header = lines[0].strip().split("\t")
        values = lines[1].strip().split("\t")
        label = filename.split("-")[1].split(".")[0]

        try:
            zall_index = header.index("ZAll")
            zall_val = re.sub(r"\([^)]*\)", "", values[zall_index].strip())
            if zall_val:
                variables[f"Zall_{label}"] = float(zall_val)
        except ValueError:
            pass

        zacc_indices = [i for i, col in enumerate(header) if col.strip()=="ZAcc"]
        for idx, zacc_index in enumerate(zacc_indices):
            val = re.sub(r"\([^)]*\)", "", values[zacc_index].strip())
            if val:
                variables[f"Zacc_{label}_m{idx+1}"] = float(val)

        try:
            zvbb_index = header.index("ZVbb")
            zvbb_val = re.sub(r"\([^)]*\)", "", values[zvbb_index].strip())
            if zvbb_val:
                variables[f"Zvbb_{label}"] = float(zvbb_val)
        except ValueError:
            pass

with open("../Data/Renorm/rho.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        label = parts[0]
        rho_v0_match = re.search(r"rho_V0\s*=\s*([\d.]+)", line)
        rho_vi_match = re.search(r"rho_Vi\s*=\s*([\d.]+)", line)
        if rho_v0_match:
            variables[f"rho_V0_{label}"] = float(rho_v0_match.group(1))
        if rho_vi_match:
            variables[f"rho_Vi_{label}"] = float(rho_vi_match.group(1))

#################################################################
# Plot configuration (unchanged)
#################################################################
inv_lat_sp = {
    "F1S": 2.785,
    "M1": 2.3833,
    "M2": 2.3833,
    "M3": 2.3833,
    "C1": 1.7848,
    "C2": 1.7848
}

marker_styles = {
    "C1": 'x',
    "C2": 'o',
    "M1": '^',
    "M2": 's',
    "M3": 'D',
    "F1S": 'v',
}

base_colors = {"C": plt.cm.Reds, "M": plt.cm.Blues}
bright_pinks = ["#43A047", "#66BB6A", "#9CCC65"]

def get_rho_prefix(ens):
    if ens.startswith("C"): return "C"
    if ens.startswith("M"): return "M"
    if ens.startswith("F"): return "F"
    raise ValueError

#################################################################
# NEW: Function to load y-values from the FitSummary CSV
#################################################################
def load_fit_values(ens, FF):
    """Reads FitSummary-FF.csv and returns central + error arrays for nsq = 1..5."""
    path = f"../Results/{ens}/Charm/FitSummary-{FF}.csv"
    if ens == 'F1S':
        m='0.259'
    elif ens in ['M1', 'M2', 'M3']:
        m='0.280'
    else:
        m='0.400'
    path2 = f"../Results/Crosschecks/AB/Crosscheck-excited-{ens}-{m}-V-3pt.csv"    
    df = pd.read_csv(path)
    df2 = pd.read_csv(path2, sep="\t")
    row = df[df["FF"] == FF].iloc[0]

    val0 = df2.iloc[0]["Value"]

    if FF in ['A0', 'V']:
        nsq_vals=[1,2,3,4,5]
    else:
        nsq_vals=[0,1,2,4,5]

    y = []
    err = []
    for k in nsq_vals:
        y.append(row[f"phys_central_nsq{k}"])
        err.append(row[f"phys_err_nsq{k}"])

    return y, err, val0

#################################################################
# NEW unified plotting function
#################################################################


def make_plot(FF, outname):
    plt.figure(figsize=(12, 8))

    for ens in Ds_mass_phys.keys():
        rho_prefix = get_rho_prefix(ens)
        rho_val = variables.get(f"rho_Vi_{rho_prefix}")
        if rho_val is None:
            continue

        # We always have exactly 5 nsq
        n_masses = 5
        prefix = ens[0]

        if prefix == "F":
            colors = bright_pinks[:n_masses]
        else:
            cmap = base_colors.get(prefix, plt.cm.Greys)
            colors = [cmap(0.5 + 0.35 * i / 4) for i in range(n_masses)]

        marker = marker_styles.get(ens, 'o')

        # Load renormalized y-values
        y_vals_raw, y_errs_raw, val0 = load_fit_values(ens, FF)

        # Compute prefactor using OLD Z and rho scheme
        zacc_val = variables.get(f"Zacc_{ens}_m1")
        zvbb_val = variables.get(f"Zvbb_{ens}")
        prefactor = rho_val * np.sqrt(zacc_val * zvbb_val)

        # val0 still read from original CSV → unchanged
        # But new val1 = Ds_mass_phys[ens][0]
        val1 = Ds_mass_phys[ens][0]

        # Compute x-axis = q² for nsq = 1..5
        x_vals = []

        ratio_lst = []

        if FF in ['A0', 'V']:
            nsq_vals=[1,2,3,4,5]
        else:
            nsq_vals=[0,1,2,4,5]

        for nsq in nsq_vals:
            valx = Ds_mass_phys[ens][nsq]
            q2 = inv_lat_sp[ens]**2 * (val0**2 + val1**2 - 2*val0*valx)
            print(val0**2,val1**2,2*val0*valx,inv_lat_sp[ens]**2,q2)
            x_vals.append(q2)
            ratio=(valx/val0)**2
            ratio_lst.append(ratio)

        # apply prefactor
        y_vals = [prefactor * y for y in y_vals_raw]
        y_errs = [prefactor * e for e in y_errs_raw]

        '''
        plt.errorbar(
            x_vals, y_vals, yerr=y_errs,
            fmt=marker,
            color=colors[0],
            capsize=3,
            label=f"{ens}"
        )
        '''

        plt.errorbar(
            ratio_lst, y_vals, yerr=y_errs,
            fmt=marker,
            color=colors[0],
            capsize=3,
            label=f"{ens}"
        )

    plt.legend()
    #plt.xlabel(r"$q^2\,[\mathrm{GeV}^2]$", fontsize=20)
    plt.xlabel(r"$(E_{D_s^*}/M_{B_s})^2$", fontsize=20)
    plt.ylabel(FF, fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outname, dpi=200, bbox_inches='tight')


#################################################################
# MAKE THE THREE NEW PLOTS
#################################################################
make_plot("V",  "V-FPlot_physical-ratio.png")
make_plot("A0", "A0-FPlot_physical-ratio.png")
make_plot("A1", "A1-FPlot_physical-ratio.png")
