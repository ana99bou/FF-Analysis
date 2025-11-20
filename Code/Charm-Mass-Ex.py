import csv
from pathlib import Path
import numpy as np

# Define the FF you want to read (currently fixed to "V")
Ens='F1S'
FF = "V"

# Your ensemble → cmass mapping
ens_dict = {
    "F1S": [0.259, 0.275],
    "M1":  [0.280, 0.340],
    "M2":  [0.280, 0.340],
    "M3":  [0.280, 0.340],
    "C1":  [0.300,0.350,0.400],
    "C2":  [0.300, 0.350, 0.400],
}

# This will store everything
# results[Ens][cmass] = [[val1], [val2], ...]
results = {}

for Ens, cmass_list in ens_dict.items():
    results[Ens] = {}
    
    for cmass in cmass_list:
        cmass_str = f"{cmass:.3f}"
        path = Path(f"../Results/{Ens}/{cmass_str}/Fit/Excited-combined-{FF}.csv")


        rows = []
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t")

            for row in reader:
                if not row:
                    continue
                if row[0].strip().startswith("#"):
                    continue
                if row[0].strip() == "nsq":   # Skip header
                    continue

        # First column = nsq, second = O00
                col1 = float(row[1])
                rows.append([col1])


                results[Ens][cmass] = rows

        # Now read JK data and fill the sublists
        jk_path = Path(f"../Results/{Ens}/{cmass_str}/Fit/Excited-combined-{FF}-jkfit.csv")

        with open(jk_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            print("Trying to read JK:", jk_path)
    
            header = None
            for row in reader:
                if not row:
                    continue
                if row[0].startswith("#"):
                    continue
        
                # Identify header row (starts with jackknife_index)
                if row[0] == "jackknife_index":
                    header = row
                    # Identify indices of O00_j columns
                    o00_cols = [i for i, name in enumerate(header) if name.startswith("O00_")]
                    continue

                # Process actual data rows
                jk_values = row
                for idx, col_index in enumerate(o00_cols):
                    val = float(jk_values[col_index])
                    results[Ens][cmass][idx].append(val)

for Ens in results:
    for cmass in results[Ens]:
        data = results[Ens][cmass]
        print(f"\n=== {Ens}  cmass={cmass} ===")
        print(f"Number of nsq lists: {len(data)}")
        
        for i, sublist in enumerate(data, start=1):
            print(f"  nsq {i}: length = {len(sublist)}")


def build_super_jackknife(results):
    """
    Build the super-jackknife blocks for all ensembles and cmasses.

    Input:
        results[Ens][cmass][q_index] = [mean, jk1, jk2, ... jkN]

    Output:
        super_jk:
            A list of length N_super
            Each element is a dict:
                super_jk[j][(Ens, cmass)][q_index] = value_j
    """

    # First collect all ensemble-cmass pairs and their JK counts
    ec_pairs = []  # list of (Ens, cmass, Nk)
    for Ens in results:
        for cmass in results[Ens]:
            # get number of jackknife samples
            # results[Ens][cmass] is a list of q² entries,
            # each entry has length Nk+1: [mean, jk1, ...]
            Nk = len(results[Ens][cmass][0]) - 1
            ec_pairs.append((Ens, cmass, Nk))

    # total number of super JK blocks
    N_super = sum(Nk for _,_,Nk in ec_pairs)

    # build cumulative index boundaries
    boundaries = []
    running = 0
    for (Ens, cmass, Nk) in ec_pairs:
        boundaries.append((Ens, cmass, running, running + Nk))
        running += Nk

    # main output structure
    super_jk = [{} for _ in range(N_super)]

    # Loop over super-jk index j
    for j in range(N_super):

        # find which ensemble-cmass block j belongs to
        for (Ens, cmass, start, end) in boundaries:
            if start <= j < end:
                active_Ens = Ens
                active_cmass = cmass
                internal_j = j - start  # internal jk index in this ensemble
                break

        # fill the data vector for all ensemble-cmass pairs
        for (Ens, cmass, Nk) in [(e,c,n) for (e,c,n) in ec_pairs]:
            super_jk[j][(Ens, cmass)] = []

            for q_idx, q_entry in enumerate(results[Ens][cmass]):

                mean = q_entry[0]
                jks = q_entry[1:]

                if Ens == active_Ens and cmass == active_cmass:
                    # use the jackknife block for this ensemble
                    value = jks[internal_j]
                else:
                    # use the central value
                    value = mean

                super_jk[j][(Ens, cmass)].append(value)

    return super_jk

mass0_all = {}

for Ens in ens_dict:
    mass0_all[Ens] = {}
    
    for cmass in ens_dict[Ens]:
        cmass_str = f"{cmass:.3f}"

        # create empty lists for nsq = 1..5
        mass0_all[Ens][cmass] = [[] for _ in range(5)]

        # -----------------------------
        # PART 1: read static Mass0
        # -----------------------------
        for nsq in range(1, 6):
            result_path = Path(f"../Data/{Ens}/2pt/Excited-comb-Ds{cmass_str}Result-{nsq}.csv")
            
            with open(result_path, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                header = None

                for row in reader:
                    if not row:
                        continue
                    if row[0].startswith("#"):
                        continue
                    
                    # header row
                    if header is None:
                        header = row
                        mass0_idx = header.index("Mass0")
                        continue
                    
                    # data row
                    static_mass0 = float(row[mass0_idx])
                    
                    # store as FIRST element
                    mass0_all[Ens][cmass][nsq - 1].append(static_mass0)
                    break   # only want first row


        # -----------------------------
        # PART 2: append m0 block values
        # -----------------------------
        for nsq in range(1, 6):
            blocks_path = Path(f"../Data/{Ens}/2pt/Excited-comb-Blocks-Ds{cmass_str}-{nsq}.csv")

            with open(blocks_path, "r") as f:
                reader = csv.reader(f, delimiter=",")  # comma-separated
                header = None

                for row in reader:
                    if not row:
                        continue

                    # header row
                    if header is None:
                        header = row
                        m0_idx = header.index("m0")
                        continue

                    # data rows: append AFTER static Mass0
                    m0_val = float(row[m0_idx])
                    mass0_all[Ens][cmass][nsq - 1].append(m0_val)

for Ens in mass0_all:
    for cmass in mass0_all[Ens]:
        data = mass0_all[Ens][cmass]
        print(f"\n=== {Ens}  cmass={cmass} ===")
        print(f"Number of nsq lists: {len(data)}")
        
        for i, sublist in enumerate(data, start=1):
            print(f"  nsq {i}: length = {len(sublist)}")

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Ens = "F1S"
cmasses = ens_dict[Ens]   # [0.259, 0.275]

nsq_vals = np.arange(1, 6)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ["blue", "red"]

for color, cmass in zip(colors, cmasses):
    ff_means = [results[Ens][cmass][i][0] for i in range(5)]       # FF mean
    mass_means = [mass0_all[Ens][cmass][i][0] for i in range(5)]   # Mass0 mean

    # CORRECT MAPPING:
    # x = nsq
    # y = Mass0
    # z = FF  (VERTICAL axis)
    ax.scatter(nsq_vals, mass_means, ff_means,
               color=color, s=80, label=f"{Ens} cmass={cmass}")

ax.set_xlabel("nsq", fontsize=12)
ax.set_ylabel("Mass0 mean", fontsize=12)
ax.set_zlabel("FF mean (O00)", fontsize=12)

ax.set_title(f"3D Plot for {Ens} (FF vertical)", fontsize=14)
ax.legend()

plt.tight_layout()
plt.savefig('Charm-Mass.pdf')
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1) Collect data for Ens = "F1S"
# ============================================================

#Ens = "F1S"
cmasses = ens_dict[Ens]      # should be [0.259, 0.275]
nsq_vals = [1, 2, 3, 4, 5]
#nsq_vals = [1, 2, 3, 5]

points = []   # will hold (cmass, nsq, mass_mean, ff_mean, ff_list)
# Exclude nsq = 4
#points = [p for p in points if p[1] != 4]

for cmass in cmasses:
    for nsq_idx, nsq in enumerate(nsq_vals):
        ff_list   = results[Ens][cmass][nsq_idx]
        mass_list = mass0_all[Ens][cmass][nsq_idx]

        ff_mean   = ff_list[0]          # first entry = mean
        mass_mean = mass_list[0]        # first entry = mean

        points.append((cmass, nsq, mass_mean, ff_mean, ff_list))

N_points = len(points)       # should be 10

# Make arrays of means
nsq_arr   = np.array([p[1] for p in points], dtype=float)      # shape (N_points,)
mass_arr  = np.array([p[2] for p in points], dtype=float)
FF_mean   = np.array([p[3] for p in points], dtype=float)

# ============================================================
# 2) Build covariance matrix from jackknife samples of FF
# ============================================================
print(points[0][4])
# Assume all FF lists have the same length = 1 (mean) + N_jk
N_jk = len(points[0][4]) - 1
for p in points:
    assert len(p[4]) - 1 == N_jk, "Inconsistent number of jackknife samples!"

# jackknife matrix JK[alpha, i] = FF_jk for sample alpha and point i
JK = np.empty((N_jk, N_points), dtype=float)

for i, p in enumerate(points):
    ff_list = p[4]
    # ff_list[1:] are the jackknife values
    JK[:, i] = np.array(ff_list[1:], dtype=float)

# Build covariance matrix using jackknife formula
Cov = np.zeros((N_points, N_points), dtype=float)
for alpha in range(N_jk):
    diff = JK[alpha, :] - FF_mean
    Cov += np.outer(diff, diff)

Cov *= (N_jk - 1) / N_jk

# Optional tiny regularisation in case of near–singular matrix
eps = 1e-12
Cov += eps * np.eye(N_points)

# Inverse covariance
Cinv = np.linalg.inv(Cov)

# ============================================================
# 3) Correlated least–squares fit to plane:
#    f(m, n) = c0 + c1*m + c2*n
# ============================================================

# Design matrix A[i] = [1, mass_i, nsq_i]
A = np.column_stack([np.ones(N_points), mass_arr, nsq_arr])  # shape (N_points, 3)


AT_Cinv = A.T @ Cinv
M = AT_Cinv @ A            # 3x3
b = AT_Cinv @ FF_mean      # 3-vector

# Parameters c = (c0, c1, c2)
c = np.linalg.solve(M, b)
# ============================================================
# JACKKNIFE PARAMETER SETS
# ============================================================

c_jk = []

for alpha in range(N_jk):
    # Jackknife FF vector (leave-one-out)
    FF_jk_alpha = JK[alpha, :]

    # Recompute M_jk and b_jk
    b_jk = AT_Cinv @ FF_jk_alpha
    c_alpha = np.linalg.solve(M, b_jk)

    c_jk.append(c_alpha)

c_jk = np.array(c_jk)   # shape (N_jk, 3)

c0, c1, c2 = c

# Parameter covariance (from normal equations)
Cov_c = np.linalg.inv(M)
c_err = np.sqrt(np.diag(Cov_c))

# chi^2
residual = FF_mean - A @ c
chi2 = residual.T @ Cinv @ residual
dof = N_points - len(c)

from scipy.stats import chi2 as chi2_dist

p_value = 1 - chi2_dist.cdf(chi2, dof)
print(f"p-value = {p_value:.6g}")

print("Fit parameters:")
print(f" c0 = {c0:.6g} ± {c_err[0]:.6g}")
print(f" c1 = {c1:.6g} ± {c_err[1]:.6g}")
print(f" c2 = {c2:.6g} ± {c_err[2]:.6g}")
print(f"chi2 = {chi2:.3f},  dof = {dof},  chi2/dof = {chi2/dof:.3f}")

# ============================================================
# 4) 3D scatter plot + fitted plane
#     axes: x = nsq, y = mass, z = FF
# ============================================================

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ["blue", "red"]

for color, cmass in zip(colors, cmasses):
    ff_means   = [results[Ens][cmass][i][0]      for i in range(5)]
    mass_means = [mass0_all[Ens][cmass][i][0]    for i in range(5)]

    ax.scatter(nsq_vals, mass_means, ff_means,
               color=color, s=80, label=f"{Ens} cmass={cmass}")

# Create plane over the range of data
nsq_grid = np.linspace(nsq_arr.min(),  nsq_arr.max(), 30)
mass_grid = np.linspace(mass_arr.min(), mass_arr.max(), 30)
NSQ, MASS = np.meshgrid(nsq_grid, mass_grid)

FF_plane = c0 + c1 * MASS + c2 * NSQ

ax.plot_surface(NSQ, MASS, FF_plane, alpha=0.3, linewidth=0, antialiased=True)

# Labels: z is vertical (FF)
ax.set_xlabel("nsq", fontsize=12)
ax.set_ylabel("Mass0 mean", fontsize=12)
ax.set_zlabel("FF mean (O00)", fontsize=12)

ax.set_title(f"Correlated plane fit for {Ens}", fontsize=14)
ax.legend()

plt.tight_layout()
plt.savefig('Charm-Mass-Fit.pdf')


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# -------------------------------------------------
# Assuming you already have:
# nsq_vals, mass_means, ff_means
# c0, c1, c2 for the fitted plane
# cmasses, Ens, results, mass0_all
# -------------------------------------------------

# You may wrap the existing plotting block into a function:
def create_3d_plot(ax):
    colors = ["blue", "red"]

    for color, cmass in zip(colors, cmasses):
        ff_means   = [results[Ens][cmass][i][0]      for i in range(5)]
        mass_means = [mass0_all[Ens][cmass][i][0]    for i in range(5)]
        ax.scatter(nsq_vals, mass_means, ff_means,
                   color=color, s=80, label=f"{Ens} cmass={cmass}")

    # Create the plane
    nsq_grid = np.linspace(nsq_vals[0], nsq_vals[-1], 30)
    mass_grid = np.linspace(min(mass_means), max(mass_means), 30)
    NSQ, MASS = np.meshgrid(nsq_grid, mass_grid)
    FF_plane = c0 + c1 * MASS + c2 * NSQ

    ax.plot_surface(NSQ, MASS, FF_plane, alpha=0.3, linewidth=0)

    ax.set_xlabel("nsq")
    ax.set_ylabel("Mass0 mean")
    ax.set_zlabel("FF mean (O00)")
    ax.set_title(f"Correlated Fit Plane — {Ens}")

# -------------------------------------------------
# Generate the PDF with multiple viewpoints
# -------------------------------------------------

view_angles = [
    (20, 30),
    (20, -60),
    (10, 120),
    (80, 0),
    (0, 0),
]

with PdfPages("3D_views.pdf") as pdf:
    for elev, azim in view_angles:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        create_3d_plot(ax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View: elev={elev}, azim={azim}")

        pdf.savefig(fig)
        plt.close(fig)

print("Saved PDF as 3D_views.pdf")


############################################

# -------------------------------------------
# Physical masses for interpolation/extrapolation
# -------------------------------------------

Eeff = {
    "F1S": {
        0: 2.107,
        1: 2.309120518,
        2: 2.491693591,
        3: 2.658876024,
        4: 2.794419221,
        5: 2.9398408
    },
    "M": {
        0: 2.107,
        1: 2.301520415,
        2: 2.476826067,
        3: 2.636953611,
        4: 2.759857922,
        5: 2.898925022
    },
    "C": {
        0: 2.107,
        1: 2.281484287,
        2: 2.43804753,
        3: 2.580365134,
        4: 2.672182955,
        5: 2.795901947
    }
}

def f_model(m, n2, c):
    """Model f(m, nsq) = c0 + c1*m + c2*n^2."""
    return c[0] + c[1]*m + c[2]*n2


def jackknife_predict_full(m_phys, nsq, c, c_jk):
    """
    Returns:
        f_mean = central prediction
        f_err  = jackknife error
        f_jk   = jackknife sample predictions
    """

    N = len(c_jk)

    # central value
    f_mean = f_model(m_phys, nsq, c)

    # jackknife predictions
    f_jk = np.array([f_model(m_phys, nsq, c_jk[alpha]) for alpha in range(N)])

    # jackknife mean of the jackknife samples
    f_bar = np.mean(f_jk)

    # jackknife variance
    var = (N - 1)/N * np.sum((f_jk - f_bar)**2)

    return f_mean, np.sqrt(var), f_jk

# Dictionary for storing full jackknife results
final_results = {}

families = {
    "F1S": Eeff["F1S"],
    #"M":   Eeff["M"],   # for M1, M2, M3 the same masses apply
    #"C":   Eeff["C"]
}

for family_name, mass_table in families.items():

    final_results[family_name] = {}

    print(f"\n=== Jackknife predictions for {family_name} ===")

    for nsq in range(1, 6):
        m_phys = mass_table[nsq]

        # get: mean, error, full jackknife blocks
        f_mean, f_err, f_jk = jackknife_predict_full(m_phys, nsq, c, c_jk)

        final_results[family_name][nsq] = {
            "mean": f_mean,
            "err":  f_err,
            "jk":   f_jk     # THIS is the crucial stored jackknife block
        }

        print(f"nsq={nsq}: f = {f_mean:.6f} ± {f_err:.6f}")
        print(f_jk.shape)

print('==================== Now improving p-value')

# ============================================================
# === UNCORRELATED FIT + SHIFTED (±1σ) DATA FITS  ============
# ============================================================

print("\n==================== Uncorrelated fit for interpolation ================")

# ------------------------------------------------------------
# 1) Compute per-point jackknife errors σ_i (uncorrelated)
# ------------------------------------------------------------
N_points = len(points)
N_jk = len(points[0][4]) - 1

sigma = np.zeros(N_points)

for i, p in enumerate(points):
    ff_list = np.array(p[4])
    mean = ff_list[0]
    jks = ff_list[1:]
    diff = jks - mean
    var = (N_jk - 1)/N_jk * np.sum(diff**2)
    sigma[i] = np.sqrt(var)

# Arrays
FF_mean = np.array([p[3] for p in points])
mass_arr = np.array([p[2] for p in points])
nsq_arr = np.array([p[1] for p in points], dtype=float)
nsq2_arr = nsq_arr**2   # actually nsq squared


# ------------------------------------------------------------
# 2) Define a function to perform uncorrelated weighted fit
# ------------------------------------------------------------
def uncorrelated_fit(FF_values, sigma, mass_arr, nsq2_arr):
    """
    Uncorrelated weighted least squares:
    χ² = Σ_i (y_i - f_i)^2 / σ_i^2
    """
    A = np.column_stack([
        np.ones_like(mass_arr),
        mass_arr,
        nsq2_arr
    ])

    w = 1.0 / sigma
    A_w = A * w[:, None]
    y_w = FF_values * w

    M = A_w.T @ A_w
    b = A_w.T @ y_w

    c = np.linalg.solve(M, b)
    Cov_c = np.linalg.inv(M)
    resid = FF_values - A @ c
    chi2 = np.sum((resid / sigma)**2)
    dof = len(FF_values) - len(c)

    return c, Cov_c, chi2, dof


# ------------------------------------------------------------
# 3) Perform fits: central, +σ shifted, -σ shifted
# ------------------------------------------------------------
c_central, Cov_c_central, chi2_central, dof_central = uncorrelated_fit(
    FF_mean, sigma, mass_arr, nsq2_arr
)

FF_plus = FF_mean + sigma
FF_minus = FF_mean - sigma

print(FF_mean)
print(sigma)

c_plus, Cov_c_plus, chi2_plus, dof_plus = uncorrelated_fit(
    FF_plus, sigma, mass_arr, nsq2_arr
)

c_minus, Cov_c_minus, chi2_minus, dof_minus = uncorrelated_fit(
    FF_minus, sigma, mass_arr, nsq2_arr
)


# ------------------------------------------------------------
# 4) Print results
# ------------------------------------------------------------
c0_central, c1_central, c2_central = c_central

print("\n=== Uncorrelated fit (central data) ===")
print(f"c0 = {c0_central:.6g}")
print(f"c1 = {c1_central:.6g}")
print(f"c2 = {c2_central:.6g}")
print(f"chi2/dof = {chi2_central/dof_central:.3f}")

print("\n=== Fit with (+1σ) shifted data ===")
print(f"c0+ = {c_plus[0]:.6g}")
print(f"c1+ = {c_plus[1]:.6g}")
print(f"c2+ = {c_plus[2]:.6g}")

print("\n=== Fit with (−1σ) shifted data ===")
print(f"c0- = {c_minus[0]:.6g}")
print(f"c1- = {c_minus[1]:.6g}")
print(f"c2- = {c_minus[2]:.6g}")

print("\n=== Parameter variations due to ±1σ shifts ===")
for i, name in enumerate(["c0", "c1", "c2"]):
    shift_plus = c_plus[i] - c_central[i]
    shift_minus = c_minus[i] - c_central[i]
    print(f"{name}: +shift = {shift_plus:.6g}, -shift = {shift_minus:.6g}")

# Optional: store uncorrelated-fit parameters
uncorr_params = {
    "central": c_central,
    "plus": c_plus,
    "minus": c_minus
}

print("\nFinished uncorrelated interpolation stability test.")
