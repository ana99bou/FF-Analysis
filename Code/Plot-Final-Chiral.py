import glob
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist


# ============================================================
# CONFIG
# ============================================================

de = 0.935  # offset in denominator of ff_model, in GeV

# Ensembles to include
active_ensembles = ["F1S","C1", "C2"]

# Form factors to plot
FF_list = ["V", "A0", "A1"]

# Optional curves    
show_continuum = False        # black dashed curve with a=0, ΔM=0

# Inverse lattice spacings 1/a in GeV
inv_lat_sp = {
    "F1S": 2.785,
    "M1":  2.3833,
    "M2":  2.3833,
    "M3":  2.3833,
    "C1":  1.7848,
    "C2":  1.7848,
}

# Ds* effective energies (nsq=0 → rest mass), all in lattice units (a E)
Ds_mass_phys = {
    "F1S": [0.756409336, 0.766627017, 0.77668534, 0.786590562, 0.796182786, 0.805801389],
    "M1":  [0.883900474, 0.902793628, 0.921202063, 0.939155484, 0.956014792, 0.973151066],
    "M2":  [0.883900474, 0.902793628, 0.921202063, 0.939155484, 0.956014792, 0.973151066],
    "M3":  [0.883900474, 0.902793628, 0.921202063, 0.939155484, 0.956014792, 0.973151066],
    "C1":  [1.180300314, 1.203099766, 1.225292858, 1.246915416, 1.266579507, 1.287189271],
    "C2":  [1.180300314, 1.203099766, 1.225292858, 1.246915416, 1.266579507, 1.287189271],
}

# Ensemble-dependent inputs (ΔMπ, a) initially in lattice units
fit_inputs = {
    "F1S": dict(MpiS=0.268,  DeltaMpi=0.052769182, a=0.359066427),
    "M1":  dict(MpiS=0.302,  DeltaMpi=0.072149182, a=0.419586288),
    "M2":  dict(MpiS=0.362,  DeltaMpi=0.111989182, a=0.419586288),
    "M3":  dict(MpiS=0.411,  DeltaMpi=0.149866182, a=0.419586288),
    "C1":  dict(MpiS=0.34,   DeltaMpi=0.096545182, a=0.560286867),
    "C2":  dict(MpiS=0.433,  DeltaMpi=0.168434182, a=0.560286867),
}

# Convert ΔMπ to physical units (GeV) and label a as a_phys (GeV^-1)
for ens, d in fit_inputs.items():
    a_inv = inv_lat_sp[ens]
    d["DeltaMpi_phys"] = d["DeltaMpi"] * a_inv        # GeV
    d["a_phys"] = d["a"]                              # already 1/a in GeV^-1

marker_styles = {
    "C1": "x",
    "C2": "o",
    "M1": "^",
    "M2": "s",
    "M3": "D",
    "F1S": "v",
}

base_colors = {"C": plt.cm.Reds, "M": plt.cm.Blues}
bright_pinks = ["#43A047", "#66BB6A", "#9CCC65"]


# ============================================================
# HELPERS
# ============================================================

def get_rho_prefix(ens):
    if ens.startswith("C"):
        return "C"
    if ens.startswith("M"):
        return "M"
    if ens.startswith("F"):
        return "F"
    raise ValueError(f"Unknown ensemble prefix for {ens}")


def nsq_list_for_FF(FF):
    if FF in ["A0", "V"]:
        return [1, 2, 3, 4, 5]
    else:  # A1
        return [0, 1, 2, 4, 5]


def choose_m_for_ens(ens):
    if ens == "F1S":
        return "0.259"
    elif ens in ["M1", "M2", "M3"]:
        return "0.280"
    else:
        return "0.400"


def load_renorm_factors():
    """
    Read ZAcc, Zvbb, rho_Vi, rho_V0 from renormalization files.
    """
    variables = {}
    # Z factors from output-*.txt
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

            zacc_indices = [i for i, col in enumerate(header) if col.strip() == "ZAcc"]
            for idx, zacc_index in enumerate(zacc_indices):
                val = re.sub(r"\([^)]*\)", "", values[zacc_index].strip())
                if val:
                    variables[f"Zacc_{label}_m{idx + 1}"] = float(val)

            try:
                zvbb_index = header.index("ZVbb")
                zvbb_val = re.sub(r"\([^)]*\)", "", values[zvbb_index].strip())
                if zvbb_val:
                    variables[f"Zvbb_{label}"] = float(zvbb_val)
            except ValueError:
                pass

    # rho factors from rho.txt
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

    return variables


def load_fit_values(ens, FF):
    """
    Read FitSummary-FF.csv and return central + error arrays for phys_nsq.
    These are form factors in lattice units (dimensionless).
    """
    path = f"../Results/{ens}/Charm/FitSummary-{FF}.csv"

    if ens == "F1S":
        m = "0.259"
    elif ens in ["M1", "M2", "M3"]:
        m = "0.280"
    else:
        m = "0.400"
    path2 = f"../Results/Crosschecks/AB/Crosscheck-excited-{ens}-{m}-V-3pt.csv"

    df = pd.read_csv(path)
    df2 = pd.read_csv(path2, sep="\t")
    row = df[df["FF"] == FF].iloc[0]

    # Bs mass in lattice units (a M_Bs)
    val0 = df2.iloc[0]["Value"]

    nsq_vals = nsq_list_for_FF(FF)

    y = []
    err = []
    for k in nsq_vals:
        y.append(row[f"phys_central_nsq{k}"])
        err.append(row[f"phys_err_nsq{k}"])

    return np.array(y), np.array(err), val0


# ============================================================
# MODEL AND CHI2 (in physical units)
# ============================================================

def ff_model(params, E_arr, ens_arr):
    """
    f(E) = [c0 + c1 ΔM_phys + c2 E + c3 E^2 + c4 a_phys + c5 a_phys^2] / (E + de)
    All mass-like quantities here are in GeV, a_phys in GeV^-1.
    """
    c0, c1, c2, c3, c4, c5 = params
    y = np.zeros_like(E_arr)

    for i, (E, ens) in enumerate(zip(E_arr, ens_arr)):
        inp = fit_inputs[ens]
        DeltaM = inp["DeltaMpi_phys"]  # GeV
        a_phys = inp["a_phys"]         # GeV^-1

        numer = (
            c0
            + c1 * DeltaM
            + c2 * E
            + c3 * E**2
            + c4 * a_phys
            + c5 * a_phys**2
        )

        y[i] = numer / (E + de)

    return y


def chi2_global(params, E_arr, ens_arr, y_mean, Cinv):
    diff = y_mean - ff_model(params, E_arr, ens_arr)
    return float(diff.T @ Cinv @ diff)


# ============================================================
# BUILD SUPER-JACKKNIFE DATA IN PHYSICAL UNITS
# ============================================================

def build_data_for_fit(FF, variables):
    """
    Returns:
        E_arr    : physical energies E_Ds*(p) in GeV
        ens_arr  : ensemble labels
        y_mean   : central values of renormalized FF
        C, Cinv  : covariance and inverse from super-jackknife
        MBs_dict : dict[ens] = physical Bs mass in GeV
    """
    nsq_phys = nsq_list_for_FF(FF)

    per_ens = {}
    total_points = 0

    for ens in active_ensembles:
        # Renormalization prefactor
        rho_prefix = get_rho_prefix(ens)
        rho = variables.get(f"rho_Vi_{rho_prefix}")
        zacc = variables.get(f"Zacc_{ens}_m1")
        zvbb = variables.get(f"Zvbb_{ens}")
        if (rho is None) or (zacc is None) or (zvbb is None):
            continue
        prefactor = rho * np.sqrt(zacc * zvbb)

        # Bs mass (lattice units) → physical
        m_choice = choose_m_for_ens(ens)
        path_bs = f"../Results/Crosschecks/AB/Crosscheck-excited-{ens}-{m_choice}-V-3pt.csv"
        if not os.path.exists(path_bs):
            continue
        MBs_latt = pd.read_csv(path_bs, sep="\t").iloc[0]["Value"]
        a_inv = inv_lat_sp[ens]
        MBs_phys = MBs_latt * a_inv  # GeV

        # JK file of unrenormalized physical FF
        path_jk = f"../Results/{ens}/Charm/PhysResults-JK-{FF}.csv"
        if not os.path.exists(path_jk):
            continue
        df_jk = pd.read_csv(path_jk)

        N_jk_e = len(df_jk)
        npts_ens = len(nsq_phys)

        Y_jk_local = np.zeros((N_jk_e, npts_ens))
        E_local = []
        ens_local = []

        for j, nsq in enumerate(nsq_phys):
            col = f"nsq{nsq}"
            raw = df_jk[col].to_numpy()        # raw FF
            Y_jk_local[:, j] = prefactor * raw  # renormalized

            # Ds* energy: aE_latt → E_phys
            aE = Ds_mass_phys[ens][nsq]
            E_phys = aE * a_inv  # GeV
            E_local.append(E_phys)
            ens_local.append(ens)

        per_ens[ens] = dict(
            jk=Y_jk_local,
            E=np.array(E_local),
            ens=np.array(ens_local, dtype=object),
            MBs=MBs_phys,
        )
        total_points += npts_ens

    # Super-jackknife construction
    N_super = sum(d["jk"].shape[0] for d in per_ens.values())
    E_arr = np.zeros(total_points)
    ens_arr = np.empty(total_points, dtype=object)
    y_central = np.zeros(total_points)

    idx0 = 0
    for ens, d in per_ens.items():
        npts = d["jk"].shape[1]
        y_central[idx0: idx0 + npts] = np.mean(d["jk"], axis=0)
        E_arr[idx0: idx0 + npts] = d["E"]
        ens_arr[idx0: idx0 + npts] = d["ens"]
        idx0 += npts

    Y_super = np.zeros((N_super, total_points))
    idx_base = {}
    cur = 0
    for ens, d in per_ens.items():
        idx_base[ens] = cur
        cur += d["jk"].shape[1]

    row = 0
    for ens, d in per_ens.items():
        jk = d["jk"]
        N_jk_e, npts_e = jk.shape
        base_e = idx_base[ens]

        for j in range(N_jk_e):
            Y_super[row] = y_central.copy()
            Y_super[row, base_e: base_e + npts_e] = jk[j]
            row += 1

    y_mean = np.mean(Y_super, axis=0)
    diffs = Y_super - y_mean
    C = (N_super - 1) / N_super * (diffs.T @ diffs)
    C += 1e-12 * np.eye(total_points)
    Cinv = np.linalg.inv(C)

    MBs_dict = {ens: d["MBs"] for ens, d in per_ens.items()}

    return E_arr, ens_arr, y_mean, C, Cinv, MBs_dict


# ============================================================
# FIT AND PLOT
# ============================================================

def fit_and_plot_FF(FF, variables):
    """
    Numerical χ² minimization in physical units, then draw per-ensemble
    curves (+ optional global & continuum curves).
    """
    E_arr, ens_arr, y_mean, C, Cinv, MBs_dict = build_data_for_fit(FF, variables)

    # Numerical minimization
    npar = 6
    p0 = np.zeros(npar)
    p0[0] = np.mean(y_mean)

    result = minimize(
        chi2_global,
        p0,
        args=(E_arr, ens_arr, y_mean, Cinv),
        method="Nelder-Mead",
    )

    if not result.success:
        print(f"WARNING: minimizer did not converge for {FF}: {result.message}")

    params_fit = result.x
    chic2 = float(result.fun)
    dof = len(y_mean) - npar
    p_value = chi2_dist.sf(chic2, dof)

    Cov_p = np.array(result.hess_inv)
    p_err = np.sqrt(np.diag(Cov_p))

    print(f"\n=== {FF} fit (numerical χ² minimum) ===")
    print("chi2:", chic2, "  dof:", dof, "  chi2/dof:", chic2 / dof, "  pvalue:", p_value)
    for i, (val, err) in enumerate(zip(params_fit, p_err)):
        print(f"c{i} = {val:.6f} ± {err:.6f}")

    ax = plt.gca()

    # Per-ensemble colored curves
    ensemble_colors = {
        "F1S": "#43A047",
        "M1": plt.cm.Blues(0.55),
        "M2": plt.cm.Blues(0.70),
        "M3": plt.cm.Blues(0.85),
        "C1": plt.cm.Reds(0.55),
        "C2": plt.cm.Reds(0.85),
    }

    MBs_for_points = np.array([MBs_dict[e] for e in ens_arr])  # GeV
    x_data = (E_arr / MBs_for_points) ** 2
    x_min, x_max = x_data.min(), x_data.max()

    for ens in active_ensembles:
        mask = (ens_arr == ens)
        if not np.any(mask):
            continue

        E_ens = E_arr[mask]
        MBs = MBs_dict[ens]
        Emin, Emax = E_ens.min(), E_ens.max()
        E_grid = np.linspace(Emin, Emax, 200)
        ens_grid = np.array([ens] * len(E_grid), dtype=object)

        y_grid = ff_model(params_fit, E_grid, ens_grid)
        x_grid = (E_grid / MBs) ** 2

        ax.plot(
            x_grid,
            y_grid,
            color=ensemble_colors.get(ens, "gray"),
            linewidth=2.0,
            label=f"{ens} fit",
        )

    # Optional: continuum a=0, ΔM=0 curve
    if show_continuum and len(MBs_dict) > 0:
        MBs_ref = np.mean(list(MBs_dict.values()))
        x_grid = np.linspace(x_min, x_max, 300)
        E_grid = MBs_ref * np.sqrt(x_grid)

        fit_inputs["PHYS"] = dict(DeltaMpi_phys=0.0, a_phys=0.0)
        ens_grid = np.array(["PHYS"] * len(E_grid), dtype=object)

        y_grid = ff_model(params_fit, E_grid, ens_grid)
        del fit_inputs["PHYS"]

        ax.plot(
            x_grid,
            y_grid,
            color="black",
            linewidth=2.0,
            linestyle="--",
            label="continuum (a=0, ΔM=0)",
        )


def make_plot(FF, variables, outname):
    """
    Plot data points (renormalized FF) for active_ensembles,
    then call fit_and_plot_FF to add curves.
    """
    plt.figure(figsize=(12, 8))

    nsq_vals = nsq_list_for_FF(FF)

    for ens in active_ensembles:
        rho_prefix = get_rho_prefix(ens)
        rho_val = variables.get(f"rho_Vi_{rho_prefix}")
        if rho_val is None:
            continue

        n_masses = len(nsq_vals)
        prefix = ens[0]
        if prefix == "F":
            colors = bright_pinks[:n_masses]
        else:
            cmap = base_colors.get(prefix, plt.cm.Greys)
            colors = [cmap(0.5 + 0.35 * i / max(1, n_masses - 1)) for i in range(n_masses)]
        marker = marker_styles.get(ens, "o")

        # Renorm prefactor
        zacc_val = variables.get(f"Zacc_{ens}_m1")
        zvbb_val = variables.get(f"Zvbb_{ens}")
        if (zacc_val is None) or (zvbb_val is None):
            continue
        prefactor = rho_val * np.sqrt(zacc_val * zvbb_val)

        # Bs mass: lattice → physical
        m_choice = choose_m_for_ens(ens)
        path_bs = f"../Results/Crosschecks/AB/Crosscheck-excited-{ens}-{m_choice}-V-3pt.csv"
        MBs_latt = pd.read_csv(path_bs, sep="\t").iloc[0]["Value"]
        a_inv = inv_lat_sp[ens]
        MBs_phys = MBs_latt * a_inv

        # central + err (unrenormalized FF)
        y_vals_raw, y_errs_raw, _ = load_fit_values(ens, FF)

        y_vals = prefactor * y_vals_raw
        y_errs = prefactor * y_errs_raw

        # x-axis: E_Ds*(p) in GeV, from Ds_mass_phys
        x_vals = []
        for nsq in nsq_vals:
            aE = Ds_mass_phys[ens][nsq]
            E_phys = aE * a_inv
            x_vals.append((E_phys / MBs_phys) ** 2)

        plt.errorbar(
            x_vals,
            y_vals,
            yerr=y_errs,
            fmt=marker,
            color=colors[0],
            capsize=3,
            label=ens,
        )

    # add fit curves on the same axes
    fit_and_plot_FF(FF, variables)

    plt.xlabel(r"$(E_{D_s^*}/M_{B_s})^2$", fontsize=20)
    plt.ylabel(FF, fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outname, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    variables = load_renorm_factors()

    for FF in FF_list:
        fname = f"{FF}-FPlot_physical-updated.png"
        print(f"\n=== Making plot for FF = {FF} → {fname}")
        make_plot(FF, variables, fname)


if __name__ == "__main__":
    main()
