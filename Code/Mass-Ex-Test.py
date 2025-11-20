import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import Ensemble as Ense
import plotly.graph_objects as go

# ============================================================
# 1. USER CONFIG
# ============================================================

ensemble = "F1S"
FF = "V"

invi = Ense.getInvSpac(ensemble)
nconf, dt, ts, L = Ense.getEns(ensemble)

ens_dict = {
    "F1S": [0.259, 0.275],
    "M1":  [0.280, 0.340],
    "M2":  [0.280, 0.340],
    "M3":  [0.280, 0.340],
    "C1":  [0.300, 0.350, 0.400],
    "C2":  [0.300, 0.350, 0.400],
}

# ============================================================
# 2. READ ALL MASSES (nsq = 0..5)
# ============================================================

mass0_all = {}

for ens_key in ens_dict:
    mass0_all[ens_key] = {}

    for cmass in ens_dict[ens_key]:
        cmass_str = f"{cmass:.3f}"

        # allocate space for nsq = 0..5 (6 sublists)
        mass0_all[ens_key][cmass] = [[] for _ in range(6)]

        # -----------------------------
        # Read static Mass0 (central)
        # -----------------------------
        for nsq in range(0, 6):
            result_path = Path(f"../Data/{ens_key}/2pt/Excited-comb-Ds{cmass_str}Result-{nsq}.csv")

            with open(result_path, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                header = None

                for row in reader:
                    if not row or row[0].startswith("#"):
                        continue

                    if header is None:
                        header = row
                        mass0_idx = header.index("Mass0")
                        continue

                    static_mass0 = float(row[mass0_idx])
                    mass0_all[ens_key][cmass][nsq].append(static_mass0)
                    break

        # -----------------------------
        # Read block jackknife values
        # -----------------------------
        for nsq in range(0, 6):
            blocks_path = Path(f"../Data/{ens_key}/2pt/Excited-comb-Blocks-Ds{cmass_str}-{nsq}.csv")

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
                    mass0_all[ens_key][cmass][nsq].append(m0_val)

# ============================================================
# 3. MASS FIT USING nsq = 0
# ============================================================

def jackknife_cov(jk_values):
    Njack = jk_values.shape[1]
    mean_jk = np.mean(jk_values, axis=1)
    diff = jk_values - mean_jk[:, None]
    return (Njack - 1) / Njack * diff @ diff.T

def fit_linear(mQ, y, cov):
    Cinv = inv(cov)
    A = np.vstack([np.ones(len(mQ)), mQ]).T
    ATA = A.T @ Cinv @ A
    ATy = A.T @ Cinv @ y
    c = np.linalg.solve(ATA, ATy)
    chi2 = (y - A @ c).T @ Cinv @ (y - A @ c)
    return c, chi2

mQ_vals = np.array(ens_dict[ensemble])  # [0.259, 0.275]

y_central = []
y_jk = []

for m in mQ_vals:
    values = mass0_all[ensemble][m][0]  # nsq = 0 = true mass
    y_central.append(values[0])
    y_jk.append(values[1:])

y_central = np.array(y_central)
y_jk = np.array(y_jk)
Njack = y_jk.shape[1]

cov = jackknife_cov(y_jk)

c_fit, chi2 = fit_linear(mQ_vals, y_central, cov)
c0, c1 = c_fit

# jackknife errors
jack_estimates = np.zeros((Njack, 2))
for i in range(Njack):
    y_jk_leave = np.delete(y_jk, i, axis=1)
    cov_jk = jackknife_cov(y_jk_leave)
    y_mean = np.mean(y_jk_leave, axis=1)
    jack_estimates[i] = fit_linear(mQ_vals, y_mean, cov_jk)[0]

c0_err, c1_err = np.sqrt((Njack - 1) * np.var(jack_estimates, axis=0))

# ============================================================
# 4. DISPERSION RELATION + EFFECTIVE ENERGIES
# ============================================================

vecs = np.array([
    [0,0,0],  # nsq=0
    [1,0,0],
    [1,1,0],
    [1,1,1],
    [2,0,0],
    [2,1,0],
])

def mass_model(mQ, c0, c1):
    return c0 + c1 * mQ

def calculate_value(a, md, p, L):
    term1 = np.sinh(md / 2)**2
    term2 = np.sin(p[0] * np.pi / L)**2
    term3 = np.sin(p[1] * np.pi / L)**2
    term4 = np.sin(p[2] * np.pi / L)**2
    return 2 * np.arcsinh(np.sqrt(term1 + term2 + term3 + term4))

a = 1.0 / invi
nsq_indices = [1,2,3,4,5]  # effective energies only

Nm = len(mQ_vals)
Nn = len(nsq_indices)

# predicted energies
E_mean = np.zeros((Nm, Nn))
E_err  = np.zeros((Nm, Nn))

E_jk = np.zeros((Nm, Nn, Njack))

for k in range(Njack):
    c0_jk, c1_jk = jack_estimates[k]

    for i, mQ in enumerate(mQ_vals):
        md_jk = mass_model(mQ, c0_jk, c1_jk)

        for j, nsq in enumerate(nsq_indices):
            E_jk[i,j,k] = calculate_value(a, md_jk, vecs[nsq], L)

for i in range(Nm):
    for j in range(Nn):
        Ejk = E_jk[i,j,:]
        mean = np.mean(Ejk)
        var  = (Njack - 1) * np.mean((Ejk - mean)**2)
        E_mean[i,j] = mean
        E_err[i,j] = np.sqrt(var)

# ============================================================
# 5. READ MEASURED EFFECTIVE ENERGIES (nsq = 1..5)
# ============================================================

E_meas_central = np.zeros((Nm, Nn))
E_meas_err     = np.zeros((Nm, Nn))

for i, mQ in enumerate(mQ_vals):
    for j, nsq in enumerate(nsq_indices):
        values = mass0_all[ensemble][mQ][nsq]
        central = values[0]
        blocks = np.array(values[1:])
        blocks = blocks[:Njack]      # match jackknife count

        mean = np.mean(blocks)
        var  = (Njack - 1) * np.mean((blocks - mean)**2)

        E_meas_central[i,j] = central
        E_meas_err[i,j] = np.sqrt(var)

# ============================================================
# 6. INTERACTIVE PLOTLY EXPORT
# ============================================================

M_grid, N_grid = np.meshgrid(mQ_vals, nsq_indices, indexing='ij')

fig = go.Figure()

fig.add_trace(go.Surface(
    x=M_grid, y=N_grid, z=E_mean,
    colorscale="Viridis", opacity=0.8
))

# measured points
fig.add_trace(go.Scatter3d(
    x=np.repeat(mQ_vals, Nn),
    y=np.tile(nsq_indices, Nm),
    z=E_meas_central.flatten(),
    mode='markers',
    marker=dict(size=6, color='red')
))

# error bars
error_x, error_y, error_z = [], [], []
for i in range(Nm):
    for j in range(Nn):
        mQ = mQ_vals[i]
        nsq = nsq_indices[j]
        zc = E_meas_central[i,j]
        err = E_meas_err[i,j]
        error_x += [mQ, mQ, None]
        error_y += [nsq, nsq, None]
        error_z += [zc-err, zc+err, None]

fig.add_trace(go.Scatter3d(
    x=error_x, y=error_y, z=error_z,
    mode='lines', line=dict(color='black', width=3)
))

fig.update_layout(
    scene=dict(
        xaxis_title="m_Q",
        yaxis_title="n_sq",
        zaxis_title="E(m_Q, n_sq)"
    ),
    width=900, height=700
)

fig.write_html("3dplot.html", include_plotlyjs="cdn")
print("\nWROTE interactive 3dplot.html")

