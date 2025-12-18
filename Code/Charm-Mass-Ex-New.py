import csv
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist
import plotly.graph_objects as go
import plotly.colors as colors


# ============================================================
# CONFIGURATION
# ============================================================

Ense = "C1"        # ensemble you want to fit
FF   = "A1"        # form factor

# list of charm masses used per ensemble
ens_dict = {
    "F1S": [0.259, 0.275],
    "M1":  [0.280, 0.340],
    "M2":  [0.280, 0.340],
    "M3":  [0.280, 0.340],
    "C1":  [0.300, 0.350, 0.400],
    "C2":  [0.300, 0.350, 0.400],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_ff_results(ens_dict, FF):
    """
    Read Excited-combined-{FF}.csv and Excited-combined-{FF}-jkfit.csv.

    results[Ens][cmass][nsq_idx] = [central, jk1, jk2, ...]
    """
    results = {}

    for ens, cmass_list in ens_dict.items():
        results[ens] = {}

        for cmass in cmass_list:
            cmass_str = f"{cmass:.3f}"
            path = Path(f"../Results/{ens}/{cmass_str}/Fit/Excited-combined-{FF}.csv")

            rows = []
            with open(path, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if not row:
                        continue
                    if row[0].strip().startswith("#"):
                        continue
                    if row[0].strip() == "nsq":
                        continue

                    # first numerical column after nsq is O00 central value
                    col_val = float(row[1])
                    rows.append([col_val])  # central value as first entry

            results[ens][cmass] = rows

            # now append jackknife blocks
            jk_path = Path(
                f"../Results/{ens}/{cmass_str}/Fit/Excited-combined-{FF}-jkfit.csv"
            )

            with open(jk_path, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                header = None
                o00_cols = None

                for row in reader:
                    if not row:
                        continue
                    if row[0].startswith("#"):
                        continue

                    if header is None and row[0] == "jackknife_index":
                        header = row
                        o00_cols = [
                            i for i, name in enumerate(header) if name.startswith("O00_")
                        ]
                        continue

                    # data rows
                    jk_values = row
                    for idx, col_index in enumerate(o00_cols):
                        val = float(jk_values[col_index])
                        results[ens][cmass][idx].append(val)

    return results


def read_mass0_blocks(ens_dict):
    """
    Read static Mass0 and block m0 values.

    mass0_all[Ens][cmass][nsq] = [Mass0_central, m0_block1, m0_block2, ...]
    """
    mass0_all = {}

    for ens, cmass_list in ens_dict.items():
        mass0_all[ens] = {}

        for cmass in cmass_list:
            cmass_str = f"{cmass:.3f}"
            # prepare 6 nsq entries per cmass
            mass0_all[ens][cmass] = [[] for _ in range(6)]

            # Part 1: static Mass0 from Result file (first data row)
            for nsq in range(6):
                result_path = Path(
                    f"../Data/{ens}/2pt/Excited-comb-Ds{cmass_str}Result-{nsq}.csv"
                )
                with open(result_path, "r") as f:
                    reader = csv.reader(f, delimiter="\t")
                    header = None
                    for row in reader:
                        if not row:
                            continue
                        if row[0].startswith("#"):
                            continue

                        if header is None:
                            header = row
                            mass0_idx = header.index("Mass0")
                            continue

                        static_mass0 = float(row[mass0_idx])
                        mass0_all[ens][cmass][nsq].append(static_mass0)
                        break  # only first data row

            # Part 2: m0 block values
            for nsq in range(6):
                blocks_path = Path(
                    f"../Data/{ens}/2pt/Excited-comb-Blocks-Ds{cmass_str}-{nsq}.csv"
                )
                with open(blocks_path, "r") as f:
                    reader = csv.reader(f, delimiter=",")
                    header = None
                    for row in reader:
                        if not row:
                            continue

                        if header is None:
                            header = row
                            m0_idx = header.index("m0")
                            continue

                        m0_val = float(row[m0_idx])
                        mass0_all[ens][cmass][nsq].append(m0_val)

    return mass0_all


def jackknife_covariance(JK, FF_mean):
    """
    Compute covariance matrix from jackknife samples.

    JK: shape (N_jk, N_points)
    """
    N_jk, N_points = JK.shape
    Cov = np.zeros((N_points, N_points))

    for alpha in range(N_jk):
        diff = JK[alpha] - FF_mean
        Cov += np.outer(diff, diff)

    Cov *= (N_jk - 1) / N_jk
    return Cov


def correlated_linear_fit(A, FF_mean, Cinv):
    """
    Solve correlated linear least squares:

      c = (A^T C^{-1} A)^{-1} A^T C^{-1} y
    """
    ATCi = A.T @ Cinv
    M = ATCi @ A
    b = ATCi @ FF_mean
    c = np.linalg.solve(M, b)

    resid = FF_mean - A @ c
    chi2 = resid.T @ Cinv @ resid
    dof = len(FF_mean) - len(c)
    pval = 1.0 - chi2_dist.cdf(chi2, dof)

    Cov_c = np.linalg.inv(M)
    c_err = np.sqrt(np.diag(Cov_c))

    return c, c_err, chi2, dof, pval, M, ATCi


def jackknife_fit_parameters(M, ATCi, JK):
    """
    Given fixed M = A^T C^{-1} A and ATCi = A^T C^{-1},
    compute jackknife parameter estimates for each JK sample.
    """
    N_jk = JK.shape[0]
    n_par = M.shape[0]
    c_jk = np.zeros((N_jk, n_par))

    for alpha in range(N_jk):
        FF_jk = JK[alpha]
        b_jk = ATCi @ FF_jk
        c_jk[alpha] = np.linalg.solve(M, b_jk)

    # jackknife variance
    c_bar = np.mean(c_jk, axis=0)
    c_var = (N_jk - 1) / N_jk * np.sum((c_jk - c_bar) ** 2, axis=0)
    c_err_jk = np.sqrt(c_var)

    return c_jk, c_err_jk


def fit_function(nsq, mass, params):
    c0, c1, c2, c3 = params
    return c0 + c1 * mass + c2 * nsq + c3 * (mass * nsq)


def get_mass_table(Ense):
    mass_table_F = {
        0: 0.756409336,
        1: 0.766627017,
        2: 0.77668534,
        3: 0.786590562,
        4: 0.796182786,
        5: 0.805801389,
    }

    mass_table_M = {
        0: 0.883900474,
        1: 0.902793628,
        2: 0.921202063,
        3: 0.939155484,
        4: 0.956014792,
        5: 0.973151066,
    }

    mass_table_C = {
        0: 1.180300314,
        1: 1.203099766,
        2: 1.225292858,
        3: 1.246915416,
        4: 1.266579507,
        5: 1.287189271,
    }

    if Ense == "F1S":
        return mass_table_F
    if Ense in ["M1", "M2", "M3"]:
        return mass_table_M
    if Ense in ["C1", "C2"]:
        return mass_table_C

    raise ValueError(f"No physical mass table for Ensemble {Ense}")


def nsq_values_for_FF(FF):
    if FF == "A1":
        return [0, 1, 2, 4, 5]
    else:
        return [1, 2, 3, 4, 5]


# ============================================================
# MAIN LOGIC
# ============================================================

# --- read inputs ---
results   = read_ff_results(ens_dict, FF)
mass0_all = read_mass0_blocks(ens_dict)

# --- build combined dataset for one ensemble (Ense) ---

cmasses   = ens_dict[Ense]
nsq_vals  = nsq_values_for_FF(FF)

points = []   # list of (nsq, mass_mean, ff_mean, ff_list)

for cmass in cmasses:
    for nsq_idx, nsq in enumerate(nsq_vals):
        ff_list   = results[Ense][cmass][nsq_idx]      # [mean, jk...]
        mass_list = mass0_all[Ense][cmass][nsq]        # [mean, jk...]

        ff_mean   = ff_list[0]
        mass_mean = mass_list[0]

        points.append((nsq, mass_mean, ff_mean, ff_list))

N_points = len(points)
N_jk     = len(points[0][3]) - 1

FF_mean  = np.array([p[2] for p in points], float)
nsq_arr  = np.array([p[0] for p in points], float)
mass_arr = np.array([p[1] for p in points], float)

# Jackknife matrix: JK[alpha, i]
JK = np.zeros((N_jk, N_points))
for i, (_, _, _, ff_list) in enumerate(points):
    JK[:, i] = np.array(ff_list[1:])

Cov  = jackknife_covariance(JK, FF_mean)
Cinv = np.linalg.inv(Cov)

# design matrix for plane with cross-term
A = np.column_stack(
    [
        np.ones(N_points),
        mass_arr,
        nsq_arr,
        mass_arr * nsq_arr,
    ]
)

# correlated linear fit
c, c_err, chi2, dof, pval, M, ATCi = correlated_linear_fit(A, FF_mean, Cinv)
c0, c1, c2, c3 = c

print("\n================ CORRELATED CENTRAL FIT (4 params) ================")
print(f"c0 = {c0:.6e} ± {c_err[0]:.6e}")
print(f"c1 = {c1:.6e} ± {c_err[1]:.6e}")
print(f"c2 = {c2:.6e} ± {c_err[2]:.6e}")
print(f"c3 = {c3:.6e} ± {c_err[3]:.6e}")
print(f"chi2/dof = {chi2:.3f}/{dof} = {chi2/dof:.3f}")
print(f"p-value   = {pval:.3f}")

# jackknife parameter estimates
c_jk, c_jk_err = jackknife_fit_parameters(M, ATCi, JK)

print("\n================ JACKKNIFE ERRORS (4 params) ======================")
print(f"c0_jk_error = {c_jk_err[0]:.6e}")
print(f"c1_jk_error = {c_jk_err[1]:.6e}")
print(f"c2_jk_error = {c_jk_err[2]:.6e}")
print(f"c3_jk_error = {c_jk_err[3]:.6e}")

# ============================================================
# PHYSICAL PREDICTIONS
# ============================================================

mass_table_phys = get_mass_table(Ense)
nsq_pred        = nsq_values_for_FF(FF)

phys_central = []
phys_err     = []
phys_jk      = np.zeros((N_jk, len(nsq_pred)))

for i, nsq_val in enumerate(nsq_pred):
    mass_val = mass_table_phys[nsq_val]

    # central
    central_val = fit_function(nsq_val, mass_val, c)
    phys_central.append(central_val)

    # jackknife values
    jk_vals = np.array(
        [fit_function(nsq_val, mass_val, c_jk[a]) for a in range(N_jk)]
    )
    phys_jk[:, i] = jk_vals

    mean_jk = np.mean(jk_vals)
    err_jk  = np.sqrt((N_jk - 1) / N_jk * np.sum((jk_vals - mean_jk) ** 2))
    phys_err.append(err_jk)

# ============================================================
# OUTPUT: CSV FILES
# ============================================================

out_dir = Path(f"../Results/{Ense}/Charm")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Fit summary with physical predictions ---
file1 = out_dir / f"FitSummary-{FF}.csv"

with open(file1, "w", newline="") as f:
    w = csv.writer(f)

    header = [
        "Ensemble", "FF", "chi2", "dof", "pvalue",
        "c0", "c0_err", "c1", "c1_err", "c2", "c2_err", "c3", "c3_err",
    ]
    for nsq in nsq_pred:
        header.append(f"phys_central_nsq{nsq}")
        header.append(f"phys_err_nsq{nsq}")
    w.writerow(header)

    row = [
        Ense, FF, chi2, dof, pval,
        c0, c_err[0], c1, c_err[1], c2, c_err[2], c3, c_err[3],
    ]
    for cval, err in zip(phys_central, phys_err):
        row.extend([cval, err])
    w.writerow(row)

print("Saved Fit Summary →", file1)

# --- Jackknife physical results ---
file2 = out_dir / f"PhysResults-JK-{FF}.csv"

with open(file2, "w", newline="") as f:
    w = csv.writer(f)
    header = ["jk_index"] + [f"nsq{nsq}" for nsq in nsq_pred]
    w.writerow(header)

    for a in range(N_jk):
        row = [a] + list(phys_jk[a, :])
        w.writerow(row)

print("Saved Jackknife Physical Results →", file2)

# ============================================================
# 3D PLOT (points + fit plane ±1σ + physical points)
# ============================================================

# grid for plane
nsq_grid  = np.linspace(nsq_arr.min(), nsq_arr.max(), 40)
mass_grid = np.linspace(mass_arr.min(), mass_arr.max(), 40)
NSQ, MASS = np.meshgrid(nsq_grid, mass_grid)

FF_plane = (
    c0
    + c1 * MASS
    + c2 * NSQ
    + c3 * MASS * NSQ
)

# jackknife planes
FF_jk_planes = np.zeros((N_jk, MASS.shape[0], MASS.shape[1]))
for a in range(N_jk):
    c0_j, c1_j, c2_j, c3_j = c_jk[a]
    FF_jk_planes[a] = (
        c0_j
        + c1_j * MASS
        + c2_j * NSQ
        + c3_j * MASS * NSQ
    )

FF_mean_plane = np.mean(FF_jk_planes, axis=0)
FF_var_plane  = (N_jk - 1) / N_jk * np.sum((FF_jk_planes - FF_mean_plane) ** 2, axis=0)
FF_err_plane  = np.sqrt(FF_var_plane)

FF_plane_plus  = FF_mean_plane + FF_err_plane
FF_plane_minus = FF_mean_plane - FF_err_plane

# separate original points by charm mass for plotting
cmasses = ens_dict[Ense]
cmass_points = {c: {"nsq": [], "mass": [], "ff": [], "err": []} for c in cmasses}

idx = 0
for cmass in cmasses:
    for _ in nsq_vals:
        nsq_val, m_val, ff_mean_val, ff_list = points[idx]
        idx += 1

        cmass_points[cmass]["nsq"].append(nsq_val)
        cmass_points[cmass]["mass"].append(m_val)
        cmass_points[cmass]["ff"].append(ff_mean_val)

        jk_vals = np.array(ff_list[1:])
        mean_jk = np.mean(jk_vals)
        err_jk  = np.sqrt((N_jk - 1) / N_jk * np.sum((jk_vals - mean_jk) ** 2))
        cmass_points[cmass]["err"].append(err_jk)

# build 3D figure
color_list = colors.qualitative.Dark24
fig = go.Figure()

# data points + error bars
for i, cmass in enumerate(cmasses):
    col = color_list[i % len(color_list)]
    data = cmass_points[cmass]

    fig.add_trace(go.Scatter3d(
        x=data["nsq"],
        y=data["mass"],
        z=data["ff"],
        mode="markers",
        name=f"{Ense}, cmass={cmass}",
        marker=dict(size=6, color=col),
    ))

    for x, y, z, e in zip(data["nsq"], data["mass"], data["ff"], data["err"]):
        fig.add_trace(go.Scatter3d(
            x=[x, x],
            y=[y, y],
            z=[z - e, z + e],
            mode="lines",
            line=dict(color=col, width=3),
            showlegend=False,
        ))

# central plane
fig.add_trace(go.Surface(
    x=NSQ,
    y=MASS,
    z=FF_plane,
    showscale=False,
    opacity=0.55,
    name="Central fit plane",
    colorscale="Viridis",
))

# +1σ and −1σ planes
fig.add_trace(go.Surface(
    x=NSQ,
    y=MASS,
    z=FF_plane_plus,
    showscale=False,
    opacity=0.35,
    name="+1σ plane",
    surfacecolor=np.zeros_like(FF_plane_plus),
    colorscale=[[0, "red"], [1, "red"]],
))

fig.add_trace(go.Surface(
    x=NSQ,
    y=MASS,
    z=FF_plane_minus,
    showscale=False,
    opacity=0.35,
    name="−1σ plane",
    surfacecolor=np.zeros_like(FF_plane_minus),
    colorscale=[[0, "blue"], [1, "blue"]],
))

# physical prediction points
phys_nsq  = nsq_pred
phys_mass = [mass_table_phys[n] for n in nsq_pred]
phys_ff   = phys_central
phys_errz = phys_err

fig.add_trace(go.Scatter3d(
    x=phys_nsq,
    y=phys_mass,
    z=phys_ff,
    mode="markers",
    name="Physical FF (Ds mass)",
    marker=dict(size=9, color="black", symbol="diamond"),
))

for x, y, z, e in zip(phys_nsq, phys_mass, phys_ff, phys_errz):
    fig.add_trace(go.Scatter3d(
        x=[x, x],
        y=[y, y],
        z=[z - e, z + e],
        mode="lines",
        line=dict(color="black", width=4),
        showlegend=False,
    ))

fig.update_layout(
    scene=dict(
        xaxis_title="nsq",
        yaxis_title="Mass0 mean",
        zaxis_title="FF mean (O00)",
    ),
    width=900,
    height=700,
)

html_path = out_dir / f"fit3D-{FF}.html"
fig.write_html(html_path)
print("Saved interactive plot as", html_path)
