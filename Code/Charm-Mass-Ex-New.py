import csv
from pathlib import Path
import numpy as np

# Define the FF you want to read (currently fixed to "V")
Ense='F1S'
FF = "A1"

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

# ============================================================
# === Build combined dataset for fitting =====================
# ============================================================

#Ense = "F1S"
cmasses = ens_dict[Ense]
if FF =='A1':
    nsq_vals=[0,1,2,4,5]    
else:
    nsq_vals = [1,2,3,4,5]
#nsq_vals = [1,2,4,5]

points = []   # list of (nsq, mass_mean, ff_mean, ff_jk)

for cmass in cmasses:
    for nsq_idx, nsq in enumerate(nsq_vals):

        ff_list   = results[Ense][cmass][nsq_idx]      # [mean, jk...]
        mass_list = mass0_all[Ense][cmass][nsq_idx]    # [mean, jk...]

        ff_mean   = ff_list[0]
        mass_mean = mass_list[0]

        # store one point
        points.append((nsq, mass_mean, ff_mean, ff_list))


# ============================================================
# === Build covariance matrix from jackknife blocks ==========
# ============================================================

N_points = len(points)
N_jk     = len(points[0][3]) - 1       # # of jackknife samples

FF_mean = np.array([p[2] for p in points], float)
nsq_arr = np.array([p[0] for p in points], float)
mass_arr = np.array([p[1] for p in points], float)

# Jackknife FF matrix JK[alpha, i]
JK = np.zeros((N_jk, N_points))
for i, (_,_,_,ff_list) in enumerate(points):
    JK[:, i] = np.array(ff_list[1:])    # jk values only


# Covariance matrix
Cov = np.zeros((N_points, N_points))
for alpha in range(N_jk):
    diff = JK[alpha] - FF_mean
    Cov += np.outer(diff, diff)

Cov *= (N_jk - 1)/N_jk
#Cov += 1e-12 * np.eye(N_points)         # stabilizer

Cinv = np.linalg.inv(Cov)


# ============================================================
# === Correlated χ² fit to: f = c0 + c1 * mass + c2 * nsq ====
# ============================================================

A = np.column_stack([
    np.ones(N_points),
    mass_arr,
    nsq_arr
])

ATCi = A.T @ Cinv
M = ATCi @ A
b = ATCi @ FF_mean
c = np.linalg.solve(M, b)

c0, c1, c2 = c
Cov_c = np.linalg.inv(M)
c_err = np.sqrt(np.diag(Cov_c))

res = FF_mean - A @ c
chi2 = res.T @ Cinv @ res
dof = N_points - 3

from scipy.stats import chi2 as chi2_dist
pval = 1 - chi2_dist.cdf(chi2, dof)

print("\n================ CORRELATED CENTRAL FIT ================")
print(f"c0 = {c0:.6e} ± {c_err[0]:.6e}")
print(f"c1 = {c1:.6e} ± {c_err[1]:.6e}")
print(f"c2 = {c2:.6e} ± {c_err[2]:.6e}")
print(f"chi2/dof = {chi2:.3f}/{dof} = {chi2/dof:.3f}")
print(f"p-value = {pval:.3f}")


# ============================================================
# === Jackknife parameter estimates ==========================
# ============================================================

c_jk = np.zeros((N_jk, 3))

for alpha in range(N_jk):
    FF_jk = JK[alpha]
    b_jk = ATCi @ FF_jk
    c_jk[alpha] = np.linalg.solve(M, b_jk)

# Jackknife variance of parameters
c_bar = np.mean(c_jk, axis=0)
c_var = (N_jk - 1)/N_jk * np.sum((c_jk - c_bar)**2, axis=0)
c_jk_err = np.sqrt(c_var)

print("\n================ JACKKNIFE ERRORS ======================")
print(f"c0_jk_error = {c_jk_err[0]:.6e}")
print(f"c1_jk_error = {c_jk_err[1]:.6e}")
print(f"c2_jk_error = {c_jk_err[2]:.6e}")

import plotly.graph_objects as go
import numpy as np

# ============================================================
# Create fine grid for the fitted plane
# ============================================================

nsq_grid = np.linspace(nsq_arr.min(), nsq_arr.max(), 40)
mass_grid = np.linspace(mass_arr.min(), mass_arr.max(), 40)
NSQ, MASS = np.meshgrid(nsq_grid, mass_grid)

FF_plane = c0 + c1 * MASS + c2 * NSQ

# ============================================================
# Separate points by charm mass for coloring
# ============================================================

cmass1 = ens_dict[Ense][0]
cmass2 = ens_dict[Ense][1]

nsq_1  = []
mass_1 = []
ff_1   = []

nsq_2  = []
mass_2 = []
ff_2   = []

for i, cmass in enumerate([cmass1, cmass2]):
    #for nsq_idx in range(5):
        #p = points[i*5 + nsq_idx]  # (nsq, mass, ff, ff_list)
    for nsq_idx in range(len(nsq_vals)):
        p = points[i * len(nsq_vals) + nsq_idx]  # (
        nsq, m, ff_mean, _ = p

        if cmass == cmass1:
            nsq_1.append(nsq)
            mass_1.append(m)
            ff_1.append(ff_mean)
        else:
            nsq_2.append(nsq)
            mass_2.append(m)
            ff_2.append(ff_mean)

# ============================================================
# Build the 3D figure
# ============================================================

fig = go.Figure()

# Data points – cmass 1
fig.add_trace(go.Scatter3d(
    x=nsq_1, y=mass_1, z=ff_1,
    mode='markers',
    name=f"{Ense}, cmass={cmass1}",
    marker=dict(size=6, color='blue')
))

# Data points – cmass 2
fig.add_trace(go.Scatter3d(
    x=nsq_2, y=mass_2, z=ff_2,
    mode='markers',
    name=f"{Ense}, cmass={cmass2}",
    marker=dict(size=6, color='red')
))

# Fitted plane
fig.add_trace(go.Surface(
    x=NSQ, y=MASS, z=FF_plane,
    showscale=False,
    opacity=0.6,
    name="Fit plane"
))

# ============================================================
# Axis labels and layout
# ============================================================

fig.update_layout(
    title=f"Correlated Fit 3D — {Ense}",
    scene=dict(
        xaxis_title="nsq",
        yaxis_title="Mass0 mean",
        zaxis_title="FF mean (O00)"
    ),
    width=900,
    height=700
)

# ============================================================
# Save as HTML
# ============================================================

fig.write_html("fit3D.html")
print("Saved interactive plot as fit3D.html")
