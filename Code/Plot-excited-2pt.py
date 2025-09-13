import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Settings
ensemble = "C1"
cmass = '0.300'

# Read in CSV with results
file = f'../Data/{ensemble}/2pt/Excited-Ds{cmass}Result-2.csv'
df = pd.read_csv(file, sep='\t', index_col=0)

# Extract central values
A1 = df.loc[0, 'Amplitude1']
A2 = df.loc[0, 'Amplitude2']
m0 = df.loc[0, 'Mass0']
m1 = df.loc[0, 'Mass1']

# Extract uncertainties
dA1 = df.loc[0, 'DeltaA1']
dA2 = df.loc[0, 'DeltaA2']
dm0 = df.loc[0, 'DeltaM0']
dm1 = df.loc[0, 'DeltaM1']


# --- Read data points ---
mass_file = f'../Data/{ensemble}/2pt/Mass-Ds{cmass}-2.csv'
df_mass = pd.read_csv(mass_file, sep='\t')

t_data = df_mass.iloc[:,0].values   # first column = t
mass_data = df_mass['EffectiveMass'].values
err_data = df_mass['Error'].values


# Define correlator and effective mass
def C2pt(t, A1, m0, A2, m1):
    return A1 * np.exp(-m0 * t) + A2 * np.exp(-m1 * t)

def E_eff(t, A1, m0, A2, m1):
    num = C2pt(t, A1, m0, A2, m1) + C2pt(t+2, A1, m0, A2, m1)
    den = 2 * C2pt(t+1, A1, m0, A2, m1)
    return np.arccosh(num / den)

# Range of t-values
t_vals = np.arange(0, 30, 1)

# Monte Carlo error propagation
Nmc = 5000
rng = np.random.default_rng(1234)
samples = []

# Keep central from fit
E_central = E_eff(t_vals, A1, m0, A2, m1)

# Monte Carlo, but drop invalid samples
samples = []
for _ in range(Nmc):
    A1s = rng.normal(A1, dA1)
    A2s = rng.normal(A2, dA2)
    m0s = rng.normal(m0, dm0)
    m1s = rng.normal(m1, dm1)
    vals = E_eff(t_vals, A1s, m0s, A2s, m1s)
    if np.all(np.isreal(vals)) and not np.any(np.isnan(vals)):
        samples.append(vals.real)

samples = np.array(samples)

E_std = samples.std(axis=0)

'''
# Plot around the central curve
plt.figure(figsize=(8,5))
plt.plot(t_vals, E_central, '-', label='E_eff (central)')
plt.fill_between(t_vals, E_central - E_std, E_central + E_std,
                 alpha=0.3, label=r'$\pm 1\sigma$ band')
plt.xlabel("t")
plt.ylabel(r"$E_{\text{eff}}(t)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Test.pdf')
'''

# --- Plot ---
plt.figure(figsize=(8,5))

# Fit curve
plt.plot(t_vals, E_central, '-', color='C0')
plt.fill_between(t_vals, E_central - E_std, E_central + E_std,
                 color='C0', alpha=0.3)

# Data points with error bars
plt.errorbar(t_data, mass_data, yerr=err_data, fmt='o', color='black',
             markersize=4, capsize=3)

plt.xlabel("t")
plt.ylabel(r"$E_{\text{eff}}(t)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Test_with_data.pdf')