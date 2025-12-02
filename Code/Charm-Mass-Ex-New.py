import csv
from pathlib import Path
import numpy as np

# Define the FF you want to read (currently fixed to "V")
Ense='F1S'
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

'''
A = np.column_stack([
    np.ones(N_points),
    mass_arr,
    nsq_arr
])
'''
# === NEW FIT: f = c0 + c1*M + c2*n + c3*(M*n) =========================

A = np.column_stack([
    np.ones(N_points),
    mass_arr,
    nsq_arr,
    mass_arr * nsq_arr
])

ATCi = A.T @ Cinv
M = ATCi @ A
b = ATCi @ FF_mean
c = np.linalg.solve(M, b)


res = FF_mean - A @ c
chi2 = res.T @ Cinv @ res
dof = N_points - 4

from scipy.stats import chi2 as chi2_dist
pval = 1 - chi2_dist.cdf(chi2, dof)

'''
c0, c1, c2 = c
Cov_c = np.linalg.inv(M)
c_err = np.sqrt(np.diag(Cov_c))

print("\n================ CORRELATED CENTRAL FIT ================")
print(f"c0 = {c0:.6e} ± {c_err[0]:.6e}")
print(f"c1 = {c1:.6e} ± {c_err[1]:.6e}")
print(f"c2 = {c2:.6e} ± {c_err[2]:.6e}")
'''

c0, c1, c2, c3 = c

Cov_c = np.linalg.inv(M)
c_err = np.sqrt(np.diag(Cov_c))

print("\n================ CORRELATED CENTRAL FIT (4 params) ================")
print(f"c0 = {c0:.6e} ± {c_err[0]:.6e}")
print(f"c1 = {c1:.6e} ± {c_err[1]:.6e}")
print(f"c2 = {c2:.6e} ± {c_err[2]:.6e}")
print(f"c3 = {c3:.6e} ± {c_err[3]:.6e}")


print(f"chi2/dof = {chi2:.3f}/{dof} = {chi2/dof:.3f}")
print(f"p-value = {pval:.3f}")


# ============================================================
# === Jackknife parameter estimates ==========================
# ============================================================

c_jk = np.zeros((N_jk, 4))

for alpha in range(N_jk):
    FF_jk = JK[alpha]
    b_jk = ATCi @ FF_jk
    c_jk[alpha] = np.linalg.solve(M, b_jk)

# Jackknife variance of parameters
c_bar = np.mean(c_jk, axis=0)
c_var = (N_jk - 1)/N_jk * np.sum((c_jk - c_bar)**2, axis=0)
c_jk_err = np.sqrt(c_var)

print("\n================ JACKKNIFE ERRORS ======================")
'''
print(f"c0_jk_error = {c_jk_err[0]:.6e}")
print(f"c1_jk_error = {c_jk_err[1]:.6e}")
print(f"c2_jk_error = {c_jk_err[2]:.6e}")
'''

print("\n================ JACKKNIFE ERRORS (4 params) ======================")
print(f"c0_jk_error = {c_jk_err[0]:.6e}")
print(f"c1_jk_error = {c_jk_err[1]:.6e}")
print(f"c2_jk_error = {c_jk_err[2]:.6e}")
print(f"c3_jk_error = {c_jk_err[3]:.6e}")


import plotly.graph_objects as go
import numpy as np

# ============================================================
# Create fine grid for the fitted plane
# ============================================================

nsq_grid = np.linspace(nsq_arr.min(), nsq_arr.max(), 40)
mass_grid = np.linspace(mass_arr.min(), mass_arr.max(), 40)
NSQ, MASS = np.meshgrid(nsq_grid, mass_grid)

#FF_plane = c0 + c1 * MASS + c2 * NSQ
# New plane including cross-term
FF_plane = (
    c0
    + c1 * MASS
    + c2 * NSQ
    + c3 * MASS * NSQ
)

# ============================================================
# Compute jackknife planes
# ============================================================

FF_jk_planes = np.zeros((N_jk, MASS.shape[0], MASS.shape[1]))

for a in range(N_jk):
    c0_j, c1_j, c2_j, c3_j = c_jk[a]
    FF_jk_planes[a] = (
        c0_j
        + c1_j * MASS
        + c2_j * NSQ
        + c3_j * MASS * NSQ
    )

# Jackknife mean and variance at each plane grid point
FF_mean_plane = np.mean(FF_jk_planes, axis=0)

FF_var_plane = (N_jk - 1)/N_jk * np.sum((FF_jk_planes - FF_mean_plane)**2, axis=0)
FF_err_plane = np.sqrt(FF_var_plane)

FF_plane_plus  = FF_mean_plane + FF_err_plane
FF_plane_minus = FF_mean_plane - FF_err_plane



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

# Compute z errors using jackknife values

#err = sqrt((N_jk - 1)/N_jk * Σ (ff_jk - mean_jk)^2)

zerr_1 = []
zerr_2 = []

for cmass in [cmass1, cmass2]:
    for nsq_idx in range(len(nsq_vals)):
        p = points[(0 if cmass == cmass1 else 1) * len(nsq_vals) + nsq_idx]
        _, _, ff_mean, ff_list = p

        jk_vals = np.array(ff_list[1:])
        mean_jk = np.mean(jk_vals)
        err_jk = np.sqrt((N_jk - 1)/N_jk * np.sum((jk_vals - mean_jk)**2))

        if cmass == cmass1:
            zerr_1.append(err_jk)
        else:
            zerr_2.append(err_jk)

def z_error_segments(x, y, z, zerr, color):
    traces = []
    for xi, yi, zi, ei in zip(x, y, z, zerr):
        traces.append(go.Scatter3d(
            x=[xi, xi],
            y=[yi, yi],
            z=[zi - ei, zi + ei],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False
        ))
    return traces



fig = go.Figure()

'''
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
'''

# Data points – cmass 1
fig.add_trace(go.Scatter3d(
    x=nsq_1, y=mass_1, z=ff_1,
    mode='markers',
    name=f"{Ense}, cmass={cmass1}",
    marker=dict(size=6, color='blue')
))

# Add z error bars (cmass 1)
for t in z_error_segments(nsq_1, mass_1, ff_1, zerr_1, color='blue'):
    fig.add_trace(t)

# Data points – cmass 2
fig.add_trace(go.Scatter3d(
    x=nsq_2, y=mass_2, z=ff_2,
    mode='markers',
    name=f"{Ense}, cmass={cmass2}",
    marker=dict(size=6, color='red')
))

# Add z error bars (cmass 2)
for t in z_error_segments(nsq_2, mass_2, ff_2, zerr_2, color='red'):
    fig.add_trace(t)


'''
# Fitted plane
fig.add_trace(go.Surface(
    x=NSQ, y=MASS, z=FF_plane,
    showscale=False,
    opacity=0.6,
    name="Fit plane"
))
'''

# Central plane
fig.add_trace(go.Surface(
    x=NSQ, y=MASS, z=FF_plane,
    showscale=False,
    opacity=0.55,
    name="Central fit plane",
    colorscale="Viridis"
))

# +1σ plane
fig.add_trace(go.Surface(
    x=NSQ, y=MASS, z=FF_plane_plus,
    showscale=False,
    opacity=0.35,
    name="+1σ plane",
    surfacecolor=np.zeros_like(FF_plane_plus),
    colorscale=[[0, "red"], [1, "red"]],
))

# -1σ plane
fig.add_trace(go.Surface(
    x=NSQ, y=MASS, z=FF_plane_minus,
    showscale=False,
    opacity=0.35,
    name="−1σ plane",
    surfacecolor=np.zeros_like(FF_plane_minus),
    colorscale=[[0, "blue"], [1, "blue"]],
))




# ============================================================
# Axis labels and layout
# ============================================================

fig.update_layout(
    #title=f"Correlated Fit 3D — {Ense}",
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

fig.write_html(f"../Results/{Ense}/Charm/fit3D-{FF}.html")
print("Saved interactive plot as fit3D.html")


# ============================================================
# === PHYSICAL MASS TABLE SELECTION ==========================
# ============================================================

mass_table_F = {
    0: 0.756409336,
    1: 0.828998994,
    2: 0.894566406,
    3: 0.954605277,
    4: 1.003281042,
    5: 1.055503647,
}

mass_table_M = {
    0: 0.884068309,
    1: 0.965686408,
    2: 1.039242255,
    3: 1.106429577,
    4: 1.157998541,
    5: 1.216349189,
}

mass_table_C = {
    0: 1.180524429,
    1: 1.278285683,
    2: 1.366006012,
    3: 1.445744696,
    4: 1.497189015,
    5: 1.566507142,
}

if Ense == "F1S":
    mass_table_phys = mass_table_F
elif Ense in ["M1","M2","M3"]:
    mass_table_phys = mass_table_M
elif Ense in ["C1","C2"]:
    mass_table_phys = mass_table_C
else:
    raise ValueError(f"No physical mass table for Ensemble {Ense}")


# ============================================================
# === nsq VALUES FOR PREDICTION ==============================
# ============================================================

if FF == "A1":
    nsq_pred = [0,1,2,4,5]
else:
    nsq_pred = [1,2,3,4,5]


# ============================================================
# === FIT FUNCTION ===========================================
# ============================================================

def fit_function(nsq, mass, params):
    c0, c1, c2, c3 = params
    return c0 + c1*mass + c2*nsq + c3*(mass*nsq)


# ============================================================
# === PHYSICAL PREDICTIONS (CENTRAL + JK) ====================
# ============================================================

phys_central = []
phys_err = []
phys_jk = np.zeros((N_jk, len(nsq_pred)))

for i, nsq_val in enumerate(nsq_pred):

    mass_val = mass_table_phys[nsq_val]

    # CENTRAL
    central_val = fit_function(nsq_val, mass_val, c)
    phys_central.append(central_val)

    # JK BLOCKS
    jk_vals = np.array([fit_function(nsq_val, mass_val, c_jk[a]) for a in range(N_jk)])
    phys_jk[:, i] = jk_vals

    # JK ERROR
    mean_jk = np.mean(jk_vals)
    err_jk = np.sqrt((N_jk - 1)/N_jk * np.sum((jk_vals - mean_jk)**2))
    phys_err.append(err_jk)


# ============================================================
# === OUTPUT DIRECTORY =======================================
# ============================================================

out_dir = Path(f"../Results/{Ense}/Charm")
out_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# === FILE 1: FIT SUMMARY INCLUDING PHYSICAL PREDICTIONS =====
# ============================================================

file1 = out_dir / f"FitSummary-{FF}.csv"

with open(file1, "w", newline="") as f:
    w = csv.writer(f)

    header = [
        "Ensemble","FF","chi2","dof","pvalue",
        "c0","c0_err","c1","c1_err","c2","c2_err","c3","c3_err"
    ]

    # Add phys. prediction columns
    for nsq in nsq_pred:
        header.append(f"phys_central_nsq{nsq}")
        header.append(f"phys_err_nsq{nsq}")

    w.writerow(header)

    row = [
        Ense, FF, chi2, dof, pval,
        c0, c_err[0], c1, c_err[1], c2, c_err[2], c3, c_err[3]
    ]

    # Add central and error
    for cval, err in zip(phys_central, phys_err):
        row.extend([cval, err])

    w.writerow(row)

print("Saved Fit Summary →", file1)


# ============================================================
# === FILE 2: JACKKNIFE RESULTS FOR PHYSICAL POINTS ==========
# ============================================================

file2 = out_dir / f"PhysResults-JK-{FF}.csv"

with open(file2, "w", newline="") as f:
    w = csv.writer(f)

    header = ["jk_index"] + [f"nsq{nsq}" for nsq in nsq_pred]
    w.writerow(header)

    for a in range(N_jk):
        row = [a] + list(phys_jk[a, :])
        w.writerow(row)

print("Saved Jackknife Physical Results →", file2)
