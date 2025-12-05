import glob
import re
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

de=0.935

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
            #print(val0**2,val1**2,2*val0*valx,inv_lat_sp[ens]**2,q2)
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
            #col = f"nsq{j+1}"
            col = f"nsq{nsq}"
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

    return E_arr, ens_arr, y_mean, C, Cinv, MBs_dict, Y_super


def ff_model(params, E_arr, ens_arr):
    c0, c1, c2, c3, c4, c5 = params

    y = np.zeros_like(E_arr)

    for i, (E, ens) in enumerate(zip(E_arr, ens_arr)):
        inp = fit_inputs[ens]
        MpiS = inp["MpiS"]         # stays unused now (no chiral logs)
        DeltaM = inp["DeltaMpi"]
        a = inp["a"]

        numer = (
              c0
            + c1 * DeltaM
            + c2 * E
            + c3 * E**2
            + c4 * a
            + c5 * a**2
        )

        y[i] = numer / (E+de)

    return y


'''
def ff_model(params, E_arr, ens_arr):
    # now 7 parameters
    #c0, c1, c2, c3, c4, c5, c6 = params

    y = np.zeros_like(E_arr)

    for i, (E, ens) in enumerate(zip(E_arr, ens_arr)):
        inp = fit_inputs[ens]
        MpiS   = inp["MpiS"]
        DeltaM = inp["DeltaMpi"]
        a      = inp["a"]

        # chiral log difference term: [δf(MpiS) - δf(MpiP)] / (4π fπ)^2
        delta_chi = (chiral_f(MpiS) - _fP) / _four_pi_f_sq

        numer = (
              c0
            + c1 * DeltaM
            + c2 * E
            + c3 * E**2
            + c4 * a
            + c5 * a**2
            + c6 * delta_chi
        )

        y[i] = numer / (E+de)

    return y
'''


def chi2_global(params, E_arr, ens_arr, y_mean, Cinv):
    diff = y_mean - ff_model(params, E_arr, ens_arr)
    return float(diff.T @ Cinv @ diff)


def build_design_matrix(E_arr, ens_arr):
    """
    Build F such that y_model = F @ params, with
    params = (c0, c1, c2, c3, c4, c5).

    y_i = [c0 + c1*ΔM + c2*E + c3*E**2 + c4*a + c5*a**2] / E
        = c0*(1/E) + c1*(ΔM/E) + c2*1 + c3*E + c4*(a/E) + c5*(a^2/E)
    """
    Np = len(E_arr)
    F = np.zeros((Np, 6))

    for i, (E, ens) in enumerate(zip(E_arr, ens_arr)):
        inp = fit_inputs[ens]
        DeltaM = inp["DeltaMpi"]
        a = inp["a"]

        F[i, 0] = 1.0 / (E+de)          # c0
        F[i, 1] = DeltaM / (E+de)       # c1
        F[i, 2] = E / (E+de)              # c2
        F[i, 3] = E                # c3
        F[i, 4] = a / (E+de)            # c4
        F[i, 5] = (a * a) / (E+de)      # c5

    return F

'''
def build_design_matrix(E_arr, ens_arr):
    """
    Build F such that y_model = F @ params, with
    params = (c0, c1, c2, c3, c4, c5, c6).

    y_i = [c0 + c1*ΔM + c2*E + c3*E**2 + c4*a + c5*a**2 + c6*Δχ] / E
         = c0*(1/E) + c1*(ΔM/E) + c2*1 + c3*E + c4*(a/E) + c5*(a^2/E)
           + c6*(Δχ/E)
    """
    Np = len(E_arr)
    F = np.zeros((Np, 7))

    for i, (E, ens) in enumerate(zip(E_arr, ens_arr)):
        inp = fit_inputs[ens]
        DeltaM = inp["DeltaMpi"]
        a      = inp["a"]
        MpiS   = inp["MpiS"]

        delta_chi = (chiral_f(MpiS) - _fP) / _four_pi_f_sq

        F[i, 0] = 1.0 / (E+de)           # c0
        F[i, 1] = DeltaM / (E+de)        # c1
        F[i, 2] = E/(E+de)               # c2
        F[i, 3] = E**2/(E+de)                 # c3
        F[i, 4] = a / (E+de)             # c4
        F[i, 5] = (a * a) / (E+de)       # c5
        #F[i, 6] = delta_chi / (E+de)     # c6

    return F
'''


def fit_and_plot_FF(FF):
    
    # build data & covariance
    E_arr, ens_arr, y_mean, C, Cinv, MBs_dict, Y_super = build_data_for_fit(FF)

    # --- correlated linear least-squares fit ---
    F = build_design_matrix(E_arr, ens_arr)

    A = F.T @ Cinv @ F
    b = F.T @ Cinv @ y_mean

    Cov_p = np.linalg.inv(A)
    params_fit = Cov_p @ b

    resid = y_mean - F @ params_fit
    chic2 = float(resid.T @ Cinv @ resid)
    dof = len(y_mean) - len(params_fit)

    from scipy.stats import chi2
    p_value = chi2.sf(chic2, dof)


    #print(f"\n=== {FF} fit (analytic χ² minimum) ===")
    print("chi2:", chic2, "  dof:", dof, "  chi2/dof:", chic2/dof, "  pvalue:", p_value)

    p_err = np.sqrt(np.diag(Cov_p))
    #print("\n=== PARAMS ===")
    for i, (val, err) in enumerate(zip(params_fit, p_err)):
        print(f"c{i} = {val:.6f} ± {err:.6f}")

    ax = plt.gca()

    # === Compute x-values of the data ===
    MBs_for_points = np.array([MBs_dict[e] for e in ens_arr])
    x_data = (E_arr / MBs_for_points)**2
    x_min = x_data.min()
    x_max = x_data.max()

    # ======================================================
    # 1) FIT CURVE PER ENSEMBLE (each in its ensemble color)
    # ======================================================

    ensemble_colors = {
        "F1S": "#43A047",   # bright green
        "M1": plt.cm.Blues(0.55),
        "M2": plt.cm.Blues(0.70),
        "M3": plt.cm.Blues(0.85),
        "C1": plt.cm.Reds(0.55),
        "C2": plt.cm.Reds(0.85),
    }

    for ens in Ds_mass_phys.keys():
        mask = (ens_arr == ens)
        if not np.any(mask):
            continue

        # Ensemble-specific E-range
        E_ens = E_arr[mask]
        MBs = MBs_dict[ens]

        Emin = E_ens.min()
        Emax = E_ens.max()
        E_grid_ens = np.linspace(Emin, Emax, 200)

        ens_grid_ens = np.array([ens] * len(E_grid_ens), dtype=object)

        # Evaluate model with the ensemble’s actual ΔM and a
        y_grid_ens = ff_model(params_fit, E_grid_ens, ens_grid_ens)

        x_grid_ens = (E_grid_ens / MBs)**2

        ax.plot(
            x_grid_ens, y_grid_ens,
            color=ensemble_colors[ens],
            linewidth=2.0,
            label=f"{ens} fit"
        )

    '''
    # ======================================================
    # 2) Black PHYSICAL CURVE (a=0, ΔM=0)
    # ======================================================

    MBs_ref = np.mean(list(MBs_dict.values()))
    x_grid_phys = np.linspace(x_min, x_max, 400)
    E_grid_phys = MBs_ref * np.sqrt(x_grid_phys)

    # Insert physical inputs:
    fit_inputs["PHYS"] = dict(MpiS=MpiP, DeltaMpi=0.0, a=0.0)

    ens_phys = np.array(["PHYS"] * len(E_grid_phys), dtype=object)
    y_phys = ff_model(params_fit, E_grid_phys, ens_phys)

    del fit_inputs["PHYS"]

    ax.plot(
        x_grid_phys, y_phys,
        color="black", linewidth=3.0,
        label="physical (a=0, ΔM=0)"
    )
    '''


make_plot("V",  "V-FPlot_physical-test.png")
make_plot("A0", "A0-FPlot_physical-test.png")
make_plot("A1", "A1-FPlot_physical-test.png")



'''
def build_design_matrix(E_arr, ens_arr):
    """
    Build F such that y_model = F @ params, with
    params = (c0, c1, c2, c3, c4, c5).

    y_i = [c0 + c1*ΔM + c2*E + c3*E**2 + c4*a + c5*a**2] / E
        = c0*(1/E) + c1*(ΔM/E) + c2*1 + c3*E + c4*(a/E) + c5*(a^2/E)
    """
    Np = len(E_arr)
    F = np.zeros((Np, 6))

    for i, (E, ens) in enumerate(zip(E_arr, ens_arr)):
        inp = fit_inputs[ens]
        DeltaM = inp["DeltaMpi"]
        a = inp["a"]

        F[i, 0] = 1.0 / E          # c0
        F[i, 1] = DeltaM / E       # c1
        F[i, 2] = 1.0              # c2
        F[i, 3] = E                # c3
        F[i, 4] = a / E            # c4
        F[i, 5] = (a * a) / E      # c5

    return F
'''
'''
def fit_and_plot_FF(FF):
    
    # build data & covariance
    E_arr, ens_arr, y_mean, C, Cinv, MBs_dict, Y_super = build_data_for_fit(FF)
    
    # initial guess for parameters [c0, c1, c2, c3, c4]
    #x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


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
    

        # --- correlated linear least-squares fit ---
    F = build_design_matrix(E_arr, ens_arr)

    # A = F^T C^{-1} F,  b = F^T C^{-1} y
    A = F.T @ Cinv @ F
    b = F.T @ Cinv @ y_mean

    Cov_p = np.linalg.inv(A)
    params_fit = Cov_p @ b

    # central chi^2
    resid = y_mean - F @ params_fit
    chi2 = float(resid.T @ Cinv @ resid)
    dof = len(y_mean) - len(params_fit)

    print(f"\n=== {FF} fit (analytic χ² minimum) ===")
    print("chi2:", chi2, "  dof:", dof, "  chi2/dof:", chi2/dof)

    # parameter errors from covariance
    p_err = np.sqrt(np.diag(Cov_p))
    print("\n=== PARAMS (from Cov_p) ===")
    for i, (val, err) in enumerate(zip(params_fit, p_err)):
        print(f"c{i} = {val:.6f} ± {err:.6f}")


    ax = plt.gca()
    ######

        # === Compute x-values of the data ===
    MBs_for_points = np.array([MBs_dict[e] for e in ens_arr])
    x_data = (E_arr / MBs_for_points)**2
    x_min = x_data.min()
    x_max = x_data.max()
    x_grid = np.linspace(x_min, x_max, 400)

    # Use some representative MBs to map x -> E for the fit curve
    MBs_ref = np.mean(list(MBs_dict.values()))
    E_grid = MBs_ref * np.sqrt(x_grid)

    # build fake ensemble array with physical inputs
    fit_inputs["PHYS"] = dict(MpiS=MpiP, DeltaMpi=0.0, a=0.0)
    ens_grid = np.array(["PHYS"] * len(E_grid), dtype=object)

    y_grid = ff_model(params_fit, E_grid, ens_grid)

    del fit_inputs["PHYS"]

    ax.plot(x_grid, y_grid, color="black", linewidth=2.0, label="global fit")
    #plt.ylim(0.8, 1.5)


    ######
    
        # --------------------------------------------------------
    #  JACKKNIFE ERROR BAND FOR THE FIT CURVE
    # --------------------------------------------------------
    # compute x-grid exactly as earlier
    MBs_for_points = np.array([MBs_dict[e] for e in ens_arr])
    x_data = (E_arr / MBs_for_points)**2
    x_min = x_data.min()
    x_max = x_data.max()
    x_grid = np.linspace(x_min, x_max, 200)

    MBs_ref = np.mean(list(MBs_dict.values()))
    E_grid = MBs_ref * np.sqrt(x_grid)

    # prepare PHYS ensemble inputs
    fit_inputs["PHYS"] = dict(MpiS=MpiP, DeltaMpi=0.0, a=0.0)
    ens_grid_phys = np.array(["PHYS"] * len(E_grid), dtype=object)

    # number of jackknife samples
    N_jk = Y_super.shape[0]   # ← exists already from build_data_for_fit

    # store all jackknife curves
    y_jk_grid = np.zeros((N_jk, len(E_grid)))

    param_jk_list = []

    # --- loop over all jackknife samples ---
    for k in range(N_jk):
        # jackknife sample output for y_mean
        y_mean_jk = Y_super[k]

        # local chi2 using that sample
        def chi2_jk(params):
            diff = y_mean_jk - ff_model(params, E_arr, ens_arr)
            return float(diff.T @ Cinv @ diff)

        # re-fit parameters to this jk sample
        res_jk = minimize(chi2_jk, params_fit, method="BFGS")
        p_jk = res_jk.x

        # evaluate model on the grid
        y_jk_grid[k] = ff_model(p_jk, E_grid, ens_grid_phys)

        res_jk = minimize(chi2_jk, params_fit, method="BFGS")
        p_jk = res_jk.x

        param_jk_list.append(p_jk)


    param_jk = np.array(param_jk_list)   # shape (N_jk, n_params)

    # jackknife mean
    p_mean = np.mean(param_jk, axis=0)

    # jackknife std: sqrt((N−1)/N * Σ (p_k − p_mean)^2)
    diffs = param_jk - p_mean
    p_std = np.sqrt((N_jk - 1) / N_jk * np.sum(diffs**2, axis=0))

    print("\n=== JACKKNIFE PARAMETER ERRORS ===")
    for i, (mean, err) in enumerate(zip(p_mean, p_std)):
        print(f"c{i} = {mean:.6f} ± {err:.6f}")


    # delete phys entry
    del fit_inputs["PHYS"]

    # --- jackknife mean and std ---
    y_mean_grid = np.mean(y_jk_grid, axis=0)
    diffs = y_jk_grid - y_mean_grid
    y_std_grid = np.sqrt((N_jk - 1) / N_jk * np.sum(diffs**2, axis=0))

    # --- plot band ---
    ax.fill_between(
        x_grid,
        y_mean_grid - y_std_grid,
        y_mean_grid + y_std_grid,
        color="black",
        alpha=0.2,
        linewidth=0,
        label="fit error band"
    )

    # --- plot central fit curve ---
    ax.plot(x_grid, y_mean_grid, color="black", linewidth=2)
    plt.ylim(0.8, 1.5)
    

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
    

    
     # === Compute x-values of the data ===
    MBs_for_points = np.array([MBs_dict[e] for e in ens_arr])
    x_data = (E_arr / MBs_for_points)**2

    # === Build grid over actual x-range used in the plot ===
    x_min = x_data.min()
    x_max = x_data.max()
    x_grid = np.linspace(x_min, x_max, 400)

    # Use a representative MBs (average or picked ensemble)
    MBs_ref = np.mean(list(MBs_dict.values()))

    # Convert x-grid back to E-grid
    E_grid = MBs_ref * np.sqrt(x_grid)

    # Build fake ensemble array for continuum
    ens_grid = np.array(["PHYS"] * len(E_grid), dtype=object)
    fit_inputs["PHYS"] = dict(MpiS=MpiP, DeltaMpi=0.0, a=0.0)

    y_grid = ff_model(params_fit, E_grid, ens_grid)

    del fit_inputs["PHYS"]

    # plot
    ax.plot(x_grid, y_grid, color="black", linewidth=2.0, label="global fit")
''' 

