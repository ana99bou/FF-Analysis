import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use("Qt5Agg")
from numpy.linalg import inv
import Ensemble as Ense

# Define the FF you want to read (currently fixed to "V")
Ens='F1S'
FF = "V"
invi=Ense.getInvSpac(Ens)
nconf,dt,ts,L= Ense.getEns(Ens)
# Your ensemble → cmass mapping
ens_dict = {
    "F1S": [0.259, 0.275],
    "M1":  [0.280, 0.340],
    "M2":  [0.280, 0.340],
    "M3":  [0.280, 0.340],
    "C1":  [0.300,0.350,0.400],
    "C2":  [0.300, 0.350, 0.400],
}

mass0_all = {}

for En in ens_dict:
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

for En in mass0_all:
    for cmass in mass0_all[Ens]:
        data = mass0_all[Ens][cmass]
        print(f"\n=== {Ens}  cmass={cmass} ===")
        print(f"Number of nsq lists: {len(data)}")
        
        for i, sublist in enumerate(data, start=1):
            print(f"  nsq {i}: length = {len(sublist)}")

print(mass0_all['F1S'][0.259])  # Example access to data

def jackknife_samples(values):
    """Return central and jackknife samples as numpy arrays."""
    central = values[0]
    jk = np.array(values[1:])
    return central, jk

def jackknife_cov(jk_values):
    """
    Compute covariance matrix from jackknife samples.
    jk_values: array with shape (Npoints, Njack)
    """
    Njack = jk_values.shape[1]
    mean_jk = np.mean(jk_values, axis=1)
    diff = jk_values - mean_jk[:, None]
    cov = (Njack - 1) / Njack * diff @ diff.T
    return cov

def fit_linear(mQ, y, cov):
    """Perform linear fit minimizing chi^2."""
    Cinv = inv(cov)

    # Build design matrix A: y = A c
    A = np.vstack([np.ones(len(mQ)), mQ]).T

    # Solve normal equations
    ATA = A.T @ Cinv @ A
    ATy = A.T @ Cinv @ y
    c = np.linalg.solve(ATA, ATy)

    # chi^2
    chi2 = (y - A @ c).T @ Cinv @ (y - A @ c)

    return c, chi2


mQ_vals = np.array([0.259, 0.275])

y_central = []
y_jk = []

for m in mQ_vals:
    # Take ONLY the first sublist
    values = mass0_all[Ens][m][0]      # <-- FIX
    
    central, jk = values[0], np.array(values[1:])
    y_central.append(central)
    y_jk.append(jk)

y_central = np.array(y_central)
y_jk = np.array(y_jk)   # shape = (2, Njack)

# ============================================
#     Construct covariance matrix
# ============================================
cov = jackknife_cov(y_jk)

# ============================================
#     Central-value linear fit
# ============================================
c_fit, chi2 = fit_linear(mQ_vals, y_central, cov)
c0, c1 = c_fit

print("Central fit:")
print("c0 =", c0)
print("c1 =", c1)
print("chi^2 =", chi2)

# ============================================
#     Jackknife error on (c0, c1)
# ============================================
Njack = y_jk.shape[1]
jack_estimates = np.zeros((Njack, 2))  # (c0, c1)

for i in range(Njack):
    # leave-one-out jackknife sample
    y_jk_leave = np.delete(y_jk, i, axis=1)
    # recompute covariance
    cov_jk = jackknife_cov(y_jk_leave)
    # jackknife central y for this subsample
    y_mean_jk = np.mean(y_jk_leave, axis=1)
    # fit
    c_jk, _ = fit_linear(mQ_vals, y_mean_jk, cov_jk)
    jack_estimates[i] = c_jk

# jackknife variance
c0_err, c1_err = np.sqrt((Njack - 1) * np.var(jack_estimates, axis=0))

print("\nJackknife uncertainties:")
print("c0 =", c0, "+/-", c0_err)
print("c1 =", c1, "+/-", c1_err)

# ============================================
#     Plot the data + fit
# ============================================
xplot = np.linspace(0.259, 0.275, 200)
#xplot = np.linspace(0., 0.400, 200)
yfit = c0 + c1 * xplot

#print(mass0_all[Ens][0.259][0][0])
plt.figure(figsize=(7,5))
plt.plot(xplot, yfit, label="Linear fit")
#plt.scatter([0.350],mass0_all[Ens][0.350][0][0],label="0.350")
#plt.errorbar([0.350],mass0_all[Ens][0.350][0][0],yerr=np.sqrt(np.diag(cov))[0],label="0.350")
plt.errorbar(mQ_vals, y_central, 
             yerr=[np.sqrt(np.diag(cov))[0], np.sqrt(np.diag(cov))[1]],
             fmt='o', label="Data points")
plt.xlabel("mQ")
plt.ylabel("mass0 F1S")
plt.legend()
plt.tight_layout()
#plt.savefig('2d-mass.png')





'''
# ============================================================
#  Second Fit Ansatz:
#  f(mQ) = mQ + c0 + c1/(2*mQ)
# ============================================================

def fit_nonlinear(mQ, y, cov):
    """Fit f = mQ + c0 + c1/(2*mQ) using correlated chi^2."""

    Cinv = inv(cov)

    # design matrix: y = A c, where columns are the derivatives wrt parameters
    # y - mQ = c0 + c1/(2*mQ)
    A = np.vstack([
        np.ones_like(mQ),          # derivative wrt c0
        1/(2*mQ)                   # derivative wrt c1
    ]).T

    # Solve normal equations
    ATA = A.T @ Cinv @ A
    ATy = A.T @ Cinv @ (y - mQ)    # move known term mQ to the LHS
    c = np.linalg.solve(ATA, ATy)

    # full model prediction
    yfit = mQ + c[0] + c[1] / (2*mQ)

    # χ²
    chi2 = (y - yfit).T @ Cinv @ (y - yfit)

    return c, chi2


# ======================
# Central-value fit
# ======================
c_fit2, chi2_2 = fit_nonlinear(mQ_vals, y_central, cov)
c0_2, c1_2 = c_fit2

print("\nNon-linear fit:")
print("c0 =", c0_2)
print("c1 =", c1_2)
print("chi^2 =", chi2_2)


# ======================
# Jackknife uncertainties
# ======================
jack_estimates_2 = np.zeros((Njack, 2))

for i in range(Njack):
    # leave-one-out
    y_jk_leave = np.delete(y_jk, i, axis=1)

    # covariance of reduced sample
    cov_jk = jackknife_cov(y_jk_leave)

    # jackknife central value
    y_mean_jk = np.mean(y_jk_leave, axis=1)

    # fit
    c_jk, _ = fit_nonlinear(mQ_vals, y_mean_jk, cov_jk)
    jack_estimates_2[i] = c_jk

# errors
c0_err_2, c1_err_2 = np.sqrt((Njack - 1) * np.var(jack_estimates_2, axis=0))

print("\nJackknife uncertainties (nonlinear fit):")
print("c0 =", c0_2, "+/-", c0_err_2)
print("c1 =", c1_2, "+/-", c1_err_2)


# ======================
# Plot the second fit
# ======================
xplot = np.linspace(0.259, 0.275, 200)
yplot = xplot + c0_2 + c1_2/(2*xplot)

plt.figure(figsize=(7,5))
plt.plot(xplot, yplot, label="Nonlinear fit")

plt.errorbar(
    mQ_vals, y_central,
    yerr=np.sqrt(np.diag(cov)),
    fmt='o', label="Data points"
)

plt.xlabel("mQ")
plt.ylabel("mass0 F1S")
plt.title("Fit: f = mQ + c0 + c1/(2 mQ)")
plt.legend()
plt.tight_layout()
#plt.savefig('2d-mass-2.png')
'''

# --- Mass model from the fit ---

# Option A: linear fit
def mass_model(mQ, c0, c1):
    return c0 + c1 * mQ

# Option B: nonlinear ansatz f = mQ + c0 + c1/(2*mQ)
# def mass_model(mQ, c0, c1):
#     return mQ + c0 + c1 / (2 * mQ)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Momentum vectors (index = nsq)
vecs = np.array([
    [0, 0, 0],  # nsq = 0 (rest)
    [1, 0, 0],  # nsq = 1
    [1, 1, 0],  # nsq = 2
    [1, 1, 1],  # nsq = 3
    [2, 0, 0],  # nsq = 4
    [2, 1, 0],  # nsq = 5
])

def calculate_value(a, md, p, L):
    # 'a' is included to match your original interface, even if not used explicitly
    term1 = np.sinh(md / 2) ** 2
    term2 = np.sin(p[0] * np.pi / L) ** 2
    term3 = np.sin(p[1] * np.pi / L) ** 2
    term4 = np.sin(p[2] * np.pi / L) ** 2
    return 2 * np.arcsinh(np.sqrt(term1 + term2 + term3 + term4))

# Assuming you already have:
# mQ_vals       : np.array([0.259, 0.275])
# c0, c1        : central fit params from your mass fit
# jack_estimates: shape (Njack, 2) with [c0_jk, c1_jk]
# inv, L        : from Ens.getInvSpac / Ens.getEns

a = 1.0 / invi
nsq_indices = [1, 2, 3, 4, 5]   # ignore nsq=0, that's just the rest mass
Nm = len(mQ_vals)
Nn = len(nsq_indices)
Njack = jack_estimates.shape[0]

# Arrays to hold central values and jackknife samples
E_central = np.zeros((Nm, Nn))
E_jk = np.zeros((Nm, Nn, Njack))

# ---- Central energies from central (c0, c1) ----
for i, mQ in enumerate(mQ_vals):
    md_central = mass_model(mQ, c0, c1)
    for j, nsq in enumerate(nsq_indices):
        p = vecs[nsq]
        E_central[i, j] = calculate_value(a, md_central, p, L)

# ---- Jackknife energies from jackknife c0,c1 ----
for k in range(Njack):
    c0_jk, c1_jk = jack_estimates[k]
    for i, mQ in enumerate(mQ_vals):
        md_jk = mass_model(mQ, c0_jk, c1_jk)
        for j, nsq in enumerate(nsq_indices):
            p = vecs[nsq]
            E_jk[i, j, k] = calculate_value(a, md_jk, p, L)

# ---- Jackknife means and errors for E ----
E_mean = np.zeros_like(E_central)
E_err = np.zeros_like(E_central)

for i in range(Nm):
    for j in range(Nn):
        Ejk = E_jk[i, j, :]
        mean_jk = np.mean(Ejk)
        var_jk = (Njack - 1) * np.mean((Ejk - mean_jk) ** 2)
        E_mean[i, j] = mean_jk
        E_err[i, j] = np.sqrt(var_jk)


print("Energies from dispersion relation with jackknife errors:\n")
for i, mQ in enumerate(mQ_vals):
    print(f"mQ = {mQ}")
    for j, nsq in enumerate(nsq_indices):
        print(f"  nsq = {nsq}:  E = {E_mean[i, j]:.8f} +/- {E_err[i, j]:.8f}")
    print()

# =====================================================================
# Measured effective energies from the dataset (sublist[1]..sublist[5])
# =====================================================================

# These arrays store measured data
E_meas_central = np.zeros((Nm, Nn))
E_meas_jk      = np.zeros((Nm, Nn, Njack))
E_meas_err     = np.zeros((Nm, Nn))

for i, mQ in enumerate(mQ_vals):

    for idx, nsq in enumerate(nsq_indices):
        # sublist index = nsq (1..5)
        #values = mass0_all[Ens][f"{mQ:.3f}"][nsq]    # <-- take the nsq-th sublist
        values = mass0_all[Ens][mQ][nsq-1]

        central = values[0]
        jk_vals = np.array(values[1:])

        # Store
        E_meas_central[i, idx] = central

        # If the jackknife sizes differ, truncate to match Njack
        if len(jk_vals) != Njack:
            jk_vals = jk_vals[:Njack]

        E_meas_jk[i, idx, :] = jk_vals

        # Jackknife estimator
        mean_jk = np.mean(jk_vals)
        var_jk = (Njack - 1) * np.mean((jk_vals - mean_jk)**2)
        E_meas_err[i, idx] = np.sqrt(var_jk)


# Prepare grid for 3D surface
M, N = np.meshgrid(mQ_vals, nsq_indices, indexing='ij')  # shape (Nm, Nn)

# ==========================================================
# Plot dispersion plane + measured data points with error bars
# ==========================================================

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Surface from model (jackknife mean)
ax.plot_surface(M, N, E_mean, alpha=0.6, color='royalblue')

# Scatter measured values
for i in range(Nm):
    for j in range(Nn):
        ax.scatter(mQ_vals[i], nsq_indices[j], E_meas_central[i, j],
                   color='red', s=40)

        # Vertical error bar drawn manually
        ax.plot(
            [mQ_vals[i], mQ_vals[i]],
            [nsq_indices[j], nsq_indices[j]],
            [E_meas_central[i, j] - E_meas_err[i, j],
             E_meas_central[i, j] + E_meas_err[i, j]],
            color='black'
        )

ax.set_xlabel(r"$m_Q$")
ax.set_ylabel(r"$n_{\mathrm{sq}}$")
ax.set_zlabel(r"$E(m_Q, n_{\mathrm{sq}})$")
ax.set_title("Dispersion plane + measured energies")

plt.tight_layout()
plt.savefig('3d-Mass.png')

import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Surface(z=E_mean, x=mQ_vals, y=nsq_indices)
])

fig.write_html("dispersion.html")


