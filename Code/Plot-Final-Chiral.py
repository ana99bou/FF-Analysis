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
    # perform global correlated fit for this FF and draw curves
    fit_and_plot_FF(FF)
    plt.legend()
    #plt.xlabel(r"$q^2\,[\mathrm{GeV}^2]$", fontsize=20)
    plt.xlabel(r"$(E_{D_s^*}/M_{B_s})^2$", fontsize=20)
    plt.ylabel(FF, fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outname, dpi=200, bbox_inches='tight')


# === Ensemble-dependent input for the fit ===
fit_inputs = {
    "F1S": dict(MpiS=0.268,  DeltaMpi=0.052769182, a=0.359066427),
    "M1":  dict(MpiS=0.302,  DeltaMpi=0.072149182, a=0.419586288),
    "M2":  dict(MpiS=0.362,  DeltaMpi=0.111989182, a=0.419586288),
    "M3":  dict(MpiS=0.411,  DeltaMpi=0.149866182, a=0.419586288),
    "C1":  dict(MpiS=0.34,   DeltaMpi=0.096545182, a=0.560286867),
    "C2":  dict(MpiS=0.433,  DeltaMpi=0.168434182, a=0.560286867),
}

MpiP = 0.13803919
f_pi = 0.1302

def chiral_f(M):
    """f(M) = -3/4 * M^2 * log(M^2)"""
    return -0.75 * (M**2) * np.log(M**2)

_fP = chiral_f(MpiP)
_four_pi_f_sq = (4.0 * np.pi * f_pi)**2

from scipy.optimize import minimize  # make sure scipy is available

def nsq_list_for_FF(FF):
    # physical nsq values used for this FF
    if FF in ['A0', 'V']:
        return [1, 2, 3, 4, 5]
    else:  # A1
        return [0, 1, 2, 4, 5]

def choose_m_for_ens(ens):
    # same choice as in load_fit_values
    if ens == 'F1S':
        return '0.259'
    elif ens in ['M1', 'M2', 'M3']:
        return '0.280'
    else:
        return '0.400'

'''
def build_data_for_fit(FF):
    """
    Build:
      E_arr[i]            : E_{Ds*} (in lattice units) of point i
      ens_arr[i]          : ensemble name of point i
      y_mean[i]           : central value (renormalized) of point i
      C, Cinv             : covariance and its inverse from jackknife
      MBs_dict[ens]       : M_{Bs} (val0) for each ensemble
    """
    nsq_phys = nsq_list_for_FF(FF)

    E_list = []
    ens_list = []
    MBs_dict = {}

    jk_blocks = []      # list of arrays with shape (N_jk, n_points_for_this_ens)
    N_jk_global = None

    for ens in Ds_mass_phys.keys():
        # rho factor
        rho_prefix = get_rho_prefix(ens)
        rho_val = variables.get(f"rho_Vi_{rho_prefix}")
        if rho_val is None:
            print(f"[fit {FF}] Missing rho for {ens}, skipping.")
            continue

        # Z factors
        zacc_val = variables.get(f"Zacc_{ens}_m1")
        zvbb_val = variables.get(f"Zvbb_{ens}")
        if zacc_val is None or zvbb_val is None:
            print(f"[fit {FF}] Missing Zacc/Zvbb for {ens}, skipping.")
            continue

        prefactor = rho_val * np.sqrt(zacc_val * zvbb_val)

        # M_Bs from the same cross-check file as before
        m_choice = choose_m_for_ens(ens)
        path_bs = f"../Results/Crosschecks/AB/Crosscheck-excited-{ens}-{m_choice}-V-3pt.csv"
        if not os.path.exists(path_bs):
            print(f"[fit {FF}] Missing Bs file for {ens}: {path_bs}")
            continue
        df_bs = pd.read_csv(path_bs, sep="\t")
        MBs = df_bs.iloc[0]["Value"]
        MBs_dict[ens] = MBs

        # jackknife file for this ens and FF
        path_jk = f"../Results/{ens}/Charm/PhysResults-JK-{FF}.csv"
        if not os.path.exists(path_jk):
            print(f"[fit {FF}] Missing jackknife file for {ens}: {path_jk}")
            continue

        df_jk = pd.read_csv(path_jk)
        N_jk = len(df_jk)

        if N_jk_global is None:
            N_jk_global = N_jk
        elif N_jk_global != N_jk:
            raise RuntimeError(
                f"Number of jackknife blocks differs: {N_jk_global} vs {N_jk} for {ens}"
            )

        # local jackknife matrix for this ensemble
        npts_ens = len(nsq_phys)
        jk_local = np.zeros((N_jk, npts_ens))

        # mapping: column 'nsq1'..'nsq5' correspond in order to nsq_phys
        for j, nsq in enumerate(nsq_phys):
            col = f"nsq{j+1}"
            if col not in df_jk.columns:
                raise KeyError(f"{col} not found in JK file for {ens}, FF={FF}")
            raw_vals = df_jk[col].to_numpy()
            jk_local[:, j] = prefactor * raw_vals

            # store kinematics once per point
            E_list.append(Ds_mass_phys[ens][nsq])
            ens_list.append(ens)

        jk_blocks.append(jk_local)

    if not jk_blocks:
        raise RuntimeError(f"No data points collected for FF={FF}")

    # concatenate JK over ensembles: shape (N_jk, N_points_total)
    Y_jk = np.concatenate(jk_blocks, axis=1)   # (N_jk, Np)
    N_jk = Y_jk.shape[0]
    Np = Y_jk.shape[1]

    # central values and covariance from jackknife
    y_mean = np.mean(Y_jk, axis=0)
    diffs = Y_jk - y_mean
    factor = (N_jk - 1) / N_jk
    C = factor * diffs.T @ diffs

    # small regularization for safety
    C_reg = C + 1e-10 * np.eye(Np)
    Cinv = np.linalg.inv(C_reg)

    E_arr = np.array(E_list)
    ens_arr = np.array(ens_list, dtype=object)

    return E_arr, ens_arr, y_mean, C, Cinv, MBs_dict
'''


def build_data_for_fit(FF):
    """
    Build:
      E_arr[i]             : Energy E_{Ds*} for each point
      ens_arr[i]           : ensemble name for each point
      y_mean[i]            : global central values
      C, Cinv              : super-jackknife covariance
      MBs_dict[ens]        : Bs masses
    """

    nsq_phys = nsq_list_for_FF(FF)

    # First pass: gather per-ensemble information
    per_ens = {}   # ens -> dict with keys:
                   #    E_pts, ens_pts, MBs, raw_jk (N_jk x npts_ens)
    total_points = 0

    for ens in Ds_mass_phys.keys():

        # Prefactor for renormalization
        rho_prefix = get_rho_prefix(ens)
        rho = variables.get(f"rho_Vi_{rho_prefix}")
        zacc = variables.get(f"Zacc_{ens}_m1")
        zvbb = variables.get(f"Zvbb_{ens}")
        if rho is None or zacc is None or zvbb is None:
            continue

        prefactor = rho * np.sqrt(zacc * zvbb)

        # M_Bs from Crosscheck file
        m_choice = choose_m_for_ens(ens)
        path_bs = f"../Results/Crosschecks/AB/Crosscheck-excited-{ens}-{m_choice}-V-3pt.csv"
        if not os.path.exists(path_bs):
            continue
        MBs = pd.read_csv(path_bs, sep="\t").iloc[0]["Value"]

        # JK file
        path_jk = f"../Results/{ens}/Charm/PhysResults-JK-{FF}.csv"
        if not os.path.exists(path_jk):
            continue
        df_jk = pd.read_csv(path_jk)

        N_jk_e = len(df_jk)
        npts_ens = len(nsq_phys)

        # Local jk matrix for ensemble e
        Y_jk_local = np.zeros((N_jk_e, npts_ens))
        E_local = []
        ens_local = []

        # Fill ensemble's jackknife block
        for j, nsq in enumerate(nsq_phys):
            col = f"nsq{j+1}"
            raw = df_jk[col].to_numpy()
            Y_jk_local[:, j] = prefactor * raw

            E_local.append(Ds_mass_phys[ens][nsq])
            ens_local.append(ens)

        per_ens[ens] = dict(
            jk=Y_jk_local,
            E=np.array(E_local),
            ens=np.array(ens_local, dtype=object),
            MBs=MBs,
        )

        total_points += npts_ens

    # -------------------------
    #  Build super-jackknife
    # -------------------------
    # total SJ length = sum of all ensembles' N_jk
    N_super = sum(len(per_ens[e]["jk"]) for e in per_ens)

    # global arrays
    E_arr = np.zeros(total_points)
    ens_arr = np.empty(total_points, dtype=object)

    # stack central values first
    y_central = np.zeros(total_points)

    # fill central values (mean over jk for each ensemble)
    idx0 = 0
    for ens, d in per_ens.items():
        npts = d["jk"].shape[1]
        y_central[idx0:idx0+npts] = np.mean(d["jk"], axis=0)
        E_arr[idx0:idx0+npts] = d["E"]
        ens_arr[idx0:idx0+npts] = d["ens"]
        idx0 += npts

    # Build Y_super: shape (N_super, total_points)
    Y_super = np.zeros((N_super, total_points))

    row = 0
    idx_base = {}  # ensemble → starting index in global vector
    cur = 0
    for ens, d in per_ens.items():
        idx_base[ens] = cur
        cur += d["jk"].shape[1]

    # For each ensemble e and each of its jackknife samples j
    for ens, d in per_ens.items():
        jk = d["jk"]
        N_jk_e, npts_e = jk.shape
        base_e = idx_base[ens]

        for j in range(N_jk_e):

            # start with central values
            Y_super[row] = y_central.copy()

            # replace ensemble e's part with its jackknife sample j
            Y_super[row, base_e:base_e+npts_e] = jk[j]

            row += 1

    # Now compute covariance from super-jackknife
    y_mean = np.mean(Y_super, axis=0)
    diffs = Y_super - y_mean
    C = (N_super - 1) / N_super * (diffs.T @ diffs)
    C += 1e-12 * np.eye(total_points)
    Cinv = np.linalg.inv(C)

    # build MBs dictionary
    MBs_dict = {ens: d["MBs"] for ens, d in per_ens.items()}

    return E_arr, ens_arr, y_mean, C, Cinv, MBs_dict



def ff_model(params, E_arr, ens_arr):
    """
    Global fit model:
    1/E * ( c0*(1 + (f(MpiS)+f(MpiP))/(4π f_pi)^2)
            + c1*ΔM_pi + c2*E + c3*E^2 + c4*a^2 )
    with ensemble-dependent MpiS, ΔM_pi, a.
    """
    c0, c1, c2, c3, c4 = params
    y = np.zeros_like(E_arr)

    for i, (E, ens) in enumerate(zip(E_arr, ens_arr)):
        inp = fit_inputs[ens]
        MpiS = inp["MpiS"]
        DeltaM = inp["DeltaMpi"]
        a = inp["a"]

        fS = chiral_f(MpiS)
        chiral_factor = 1.0 + (fS + _fP) / _four_pi_f_sq

        numer = (
            c0 * chiral_factor
            + c1 * DeltaM
            + c2 * E
            + c3 * E**2
            + c4 * a**2
        )
        y[i] = numer / E

    return y

def chi2_global(params, E_arr, ens_arr, y_mean, Cinv):
    diff = y_mean - ff_model(params, E_arr, ens_arr)
    return float(diff.T @ Cinv @ diff)

def fit_and_plot_FF(FF):
    
    # build data & covariance
    E_arr, ens_arr, y_mean, C, Cinv, MBs_dict = build_data_for_fit(FF)
    
    # initial guess for parameters [c0, c1, c2, c3, c4]
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    res = minimize(
        chi2_global, x0,
        args=(E_arr, ens_arr, y_mean, Cinv),
        method="BFGS"
    )

    print(f"\n=== {FF} fit ===")
    print("success:", res.success)
    print("chi2:", res.fun, "  dof:", len(y_mean) - len(x0))
    print("params:", res.x)

    params_fit = res.x
    ax = plt.gca()

    '''
    # plot one smooth curve per ensemble using the ensemble's inputs
    for ens in Ds_mass_phys.keys():
        mask = (ens_arr == ens)
        if not np.any(mask):
            continue

        E_ens = E_arr[mask]
        Emin, Emax = E_ens.min(), E_ens.max()
        E_grid = np.linspace(Emin, Emax, 200)
        ens_grid = np.array([ens] * len(E_grid), dtype=object)
        y_grid = ff_model(params_fit, E_grid, ens_grid)

        # convert to the x-variable you plot: (E_Ds*/M_Bs)^2
        MBs = MBs_dict[ens]
        x_grid = (E_grid / MBs)**2

        prefix = ens[0]
        if prefix == "F":
            color = bright_pinks[0]
        else:
            cmap = base_colors.get(prefix, plt.cm.Greys)
            color = cmap(0.5)

        ax.plot(x_grid, y_grid, linestyle='-', color=color)
    '''
        
    # === Plot ONE single global curve ===
    # physical-continuum reference inputs
    MpiS_phys = MpiP         # = M_pi^P
    DeltaM_phys = 0.0
    a_phys = 0.0

    # global E-range from the data
    E_min = E_arr.min()
    E_max = E_arr.max()
    E_grid = np.linspace(E_min, E_max, 300)

    # build fake ensemble array with physical inputs
    ens_grid = np.array(["PHYS"] * len(E_grid), dtype=object)

    # temporary data structure for PHYS inputs
    temp = dict(MpiS=MpiS_phys, DeltaMpi=DeltaM_phys, a=a_phys)
    fit_inputs["PHYS"] = temp

    y_grid = ff_model(params_fit, E_grid, ens_grid)

    # remove temporary entry again
    del fit_inputs["PHYS"]

    # convert to your plotted x-axis: (E/M_Bs)^2
    # choose ANY Bs mass (they are just normalization), e.g. average
    MBs_avg = np.mean(list(MBs_dict.values()))
    x_grid = (E_grid / MBs_avg)**2

    ax = plt.gca()
    ax.plot(x_grid, y_grid, color="black", linewidth=2.0, label="global fit")



make_plot("V",  "V-FPlot_physical-test.png")
make_plot("A0", "A0-FPlot_physical-test.png")
#make_plot("A1", "A1-FPlot_physical-test.png")
