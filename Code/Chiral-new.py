import glob
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

# ============================================================
# CONFIG
# ============================================================

active_ensembles = ["F1S", "M3", "C1", "C2"]
FF_list = ["V", "A0", "A1"]
show_continuum = True

# Physical constants
M_pi_phys = 0.135  # physical pion mass in GeV

# Inverse lattice spacings 1/a in GeV
inv_lat_sp = {
    "F1S": 2.785, "M1": 2.3833, "M2": 2.3833, "M3": 2.3833,
    "C1": 1.7848, "C2": 1.7848,
}

# Ds* effective energies (lattice units)
Ds_mass_phys = {
    "F1S": [0.756409336, 0.766627017, 0.77668534, 0.786590562, 0.796182786, 0.805801389],
    "M1":  [0.883900474, 0.902793628, 0.921202063, 0.939155484, 0.956014792, 0.973151066],
    "M2":  [0.883900474, 0.902793628, 0.921202063, 0.939155484, 0.956014792, 0.973151066],
    "M3":  [0.883900474, 0.902793628, 0.921202063, 0.939155484, 0.956014792, 0.973151066],
    "C1":  [1.180300314, 1.203099766, 1.225292858, 1.246915416, 1.266579507, 1.287189271],
    "C2":  [1.180300314, 1.203099766, 1.225292858, 1.246915416, 1.266579507, 1.287189271],
}

# Ensemble inputs
fit_inputs = {
    "F1S": dict(MpiS=0.268, DeltaMpi=0.052769182, a=0.359066427),
    "M1":  dict(MpiS=0.302, DeltaMpi=0.072149182, a=0.419586288),
    "M2":  dict(MpiS=0.362, DeltaMpi=0.111989182, a=0.419586288),
    "M3":  dict(MpiS=0.411, DeltaMpi=0.149866182, a=0.419586288),
    "C1":  dict(MpiS=0.34,  DeltaMpi=0.096545182, a=0.560286867),
    "C2":  dict(MpiS=0.433, DeltaMpi=0.168434182, a=0.560286867),
}

for ens, d in fit_inputs.items():
    a_inv = inv_lat_sp[ens]
    d["DeltaMpi_phys"] = d["DeltaMpi"] * a_inv  # GeV
    d["MpiS_phys"] = d["MpiS"] * a_inv  # GeV (simulated pion mass)
    d["a_phys"] = 1.0 / a_inv  # GeV

marker_styles = {"C1": "x", "C2": "o", "M1": "^", "M2": "s", "M3": "D", "F1S": "v"}

# ============================================================
# HELPERS
# ============================================================

def get_rho_prefix(ens):
    return ens[0]

def nsq_list_for_FF(FF):
    return [1, 2, 3, 4, 5] if FF in ["A0", "V"] else [0, 1, 2, 4, 5]

def choose_m_for_ens(ens):
    if ens == "F1S":
        return "0.259"
    elif ens in ["M1", "M2", "M3"]:
        return "0.280"
    return "0.400"

def load_renorm_factors():
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

# ============================================================
# SIMPLIFIED FIT FUNCTION (NO CHIRAL LOGS)
# ============================================================

def fit_function(params, E_K, M_pi_sim, a, Lambda, Delta_X):
    """
    Simplified fit function WITHOUT chiral logarithms:
    
    f_X(M_π, E_K, a²) = [Λ / (E_K + Δ_X)] × 
                        [c_X,0 +
                         c_X,1 × ΔM_π²/Λ² +
                         c_X,2 × E_K/Λ +
                         c_X,3 × E_K²/Λ² +
                         c_X,4 × (aΛ)²]
    
    Parameters:
    -----------
    params : array [c_X0, c_X1, c_X2, c_X3, c_X4]
        Fit coefficients
        c_X0: normalization
        c_X1: analytic chiral correction (M_π² dependence)
        c_X2: linear energy dependence
        c_X3: quadratic energy dependence  
        c_X4: discretization correction
    E_K : float
        Energy of daughter meson (Ds*) in GeV
    M_pi_sim : float
        Simulated pion mass in GeV
    a : float
        Lattice spacing in GeV^-1
    Lambda : float
        Scale parameter in GeV (fixed)
    Delta_X : float
        Pole offset in GeV (form-factor dependent)
    """
    c_X0, c_X1, c_X2, c_X3, c_X4 = params
    
    # Pole structure (gives 1/E curvature)
    pole_factor = Lambda / (E_K + Delta_X)
    
    # ΔM_π² = (M_π^sim)² - (M_π^phys)²
    DeltaM_pi_sq = M_pi_sim**2 - M_pi_phys**2
    
    # Polynomial expansion in dimensionless variables
    expansion = (
        c_X0 +
        c_X1 * (DeltaM_pi_sq / Lambda**2) +
        c_X2 * (E_K / Lambda) +
        c_X3 * (E_K / Lambda)**2 +
        c_X4 * (a * Lambda)**2
    )
    
    return pole_factor * expansion


def evaluate_model(params, E_K_arr, M_pi_arr, a_arr, Lambda, Delta_X):
    """Evaluate model for arrays of data."""
    return np.array([fit_function(params, E, M, a, Lambda, Delta_X)
                     for E, M, a in zip(E_K_arr, M_pi_arr, a_arr)])


# ============================================================
# DATA LOADING
# ============================================================

def build_jackknife_samples(FF, variables):
    """Build jackknife samples for all ensembles."""
    nsq_phys = nsq_list_for_FF(FF)
    ensemble_data = {}
    
    for ens in active_ensembles:
        rho_prefix = get_rho_prefix(ens)
        rho = variables.get(f"rho_Vi_{rho_prefix}")
        zacc = variables.get(f"Zacc_{ens}_m1")
        zvbb = variables.get(f"Zvbb_{ens}")
        
        if (rho is None) or (zacc is None) or (zvbb is None):
            print(f"Warning: Missing renormalization for {ens}")
            continue
            
        prefactor = rho * np.sqrt(zacc * zvbb)
        
        m_choice = choose_m_for_ens(ens)
        path_bs = f"../Results/Crosschecks/AB/Crosscheck-excited-{ens}-{m_choice}-V-3pt.csv"
        if not os.path.exists(path_bs):
            print(f"Warning: Missing Bs mass file for {ens}")
            continue
            
        MBs_latt = pd.read_csv(path_bs, sep="\t").iloc[0]["Value"]
        a_inv = inv_lat_sp[ens]
        MBs_phys = MBs_latt * a_inv
        
        path_jk = f"../Results/{ens}/Charm/PhysResults-JK-{FF}.csv"
        if not os.path.exists(path_jk):
            print(f"Warning: Missing jackknife file for {ens}")
            continue
            
        df_jk = pd.read_csv(path_jk)
        
        jk_samples = []
        E_K_vals = []
        M_pi_vals = []
        a_vals = []
        q2_vals = []
        
        for nsq in nsq_phys:
            col = f"nsq{nsq}"
            if col not in df_jk.columns:
                continue
                
            jk_samples.append(prefactor * df_jk[col].to_numpy())
            
            # Energy of Ds* in GeV
            aE = Ds_mass_phys[ens][nsq]
            E_phys = aE * a_inv
            E_K_vals.append(E_phys)
            
            # Also compute q² for plotting
            q2_vals.append((E_phys / MBs_phys)**2)
            
            # Simulated pion mass
            M_pi_vals.append(fit_inputs[ens]["MpiS_phys"])
            
            # Lattice spacing
            a_vals.append(fit_inputs[ens]["a_phys"])
        
        if len(jk_samples) == 0:
            continue
            
        ensemble_data[ens] = {
            'jk_samples': np.array(jk_samples).T,
            'E_K': np.array(E_K_vals),
            'M_pi': np.array(M_pi_vals),
            'a': np.array(a_vals),
            'q2': np.array(q2_vals),
            'MBs': MBs_phys,
        }
    
    return ensemble_data


def create_combined_data(ensemble_data):
    """Combine data with proper super-jackknife."""
    n_total = sum(len(data['E_K']) for data in ensemble_data.values())
    jk_counts = [data['jk_samples'].shape[0] for data in ensemble_data.values()]
    N_jk_total = sum(jk_counts)
    
    print(f"  Super-jackknife: {len(ensemble_data)} ensembles, {jk_counts} blocks, total {N_jk_total}")
    
    E_K_all = np.zeros(n_total)
    M_pi_all = np.zeros(n_total)
    a_all = np.zeros(n_total)
    q2_all = np.zeros(n_total)
    ens_labels = []
    
    jk_combined = np.zeros((N_jk_total, n_total))
    y_central = np.zeros(n_total)
    
    # Collect data
    idx = 0
    for ens, data in ensemble_data.items():
        n_pts = len(data['E_K'])
        E_K_all[idx:idx+n_pts] = data['E_K']
        M_pi_all[idx:idx+n_pts] = data['M_pi']
        a_all[idx:idx+n_pts] = data['a']
        q2_all[idx:idx+n_pts] = data['q2']
        ens_labels.extend([ens] * n_pts)
        y_central[idx:idx+n_pts] = np.mean(data['jk_samples'], axis=0)
        idx += n_pts
    
    # Build super-jackknife
    jk_row = 0
    idx = 0
    for ens, data in ensemble_data.items():
        N_jk_ens = data['jk_samples'].shape[0]
        n_pts = data['jk_samples'].shape[1]
        
        for jk in range(N_jk_ens):
            jk_combined[jk_row, :] = y_central.copy()
            jk_combined[jk_row, idx:idx+n_pts] = data['jk_samples'][jk, :]
            jk_row += 1
        
        idx += n_pts
    
    return {
        'E_K': E_K_all,
        'M_pi': M_pi_all,
        'a': a_all,
        'q2': q2_all,
        'ens_labels': ens_labels,
        'jk_samples': jk_combined,
        'y_central': y_central
    }


# ============================================================
# FITTING
# ============================================================

def fit_with_jackknife(data_dict, Lambda, Delta_X):
    """Fit with error propagation."""
    N_jk = data_dict['jk_samples'].shape[0]
    y_mean = np.mean(data_dict['jk_samples'], axis=0)
    diff = data_dict['jk_samples'] - y_mean
    cov = (N_jk - 1) / N_jk * (diff.T @ diff)
    y_err = np.sqrt(np.diag(cov))
    
    print(f"\n  Data summary:")
    print(f"    Points: {len(y_mean)}")
    print(f"    y range: [{y_mean.min():.3f}, {y_mean.max():.3f}]")
    print(f"    E_K range: [{data_dict['E_K'].min():.3f}, {data_dict['E_K'].max():.3f}] GeV")
    print(f"    M_π range: [{data_dict['M_pi'].min():.3f}, {data_dict['M_pi'].max():.3f}] GeV")
    print(f"    Mean error: {y_err.mean():.4f} ({100*y_err.mean()/y_mean.mean():.1f}%)")
    
    # Use diagonal weights (more stable)
    weights = 1.0 / y_err
    
    def residuals(params):
        y_model = evaluate_model(params, data_dict['E_K'], data_dict['M_pi'],
                                 data_dict['a'], Lambda, Delta_X)
        return (y_mean - y_model) * weights
    
    # Initial guess
    E_mean = np.mean(data_dict['E_K'])
    y_mean_val = np.mean(y_mean)
    c_X0_guess = y_mean_val * (E_mean + Delta_X) / Lambda
    
    p0 = np.array([c_X0_guess, 0.0, 0.0, 0.0, 0.0])
    
    print(f"\n  Fit parameters:")
    print(f"    Lambda = {Lambda:.3f} GeV (fixed)")
    print(f"    Delta_X = {Delta_X:.3f} GeV (fixed)")
    print(f"  Initial guess: c_X0={p0[0]:.3f}, c_X1={p0[1]:.3f}, c_X2={p0[2]:.3f}, c_X3={p0[3]:.3f}, c_X4={p0[4]:.3f}")
    
    result = least_squares(residuals, p0, method='lm', ftol=1e-10, xtol=1e-10,
                          max_nfev=10000, verbose=0)
    
    if not result.success:
        print(f"  WARNING: Fit did not converge properly!")
    
    params_central = result.x
    y_model = evaluate_model(params_central, data_dict['E_K'], data_dict['M_pi'],
                            data_dict['a'], Lambda, Delta_X)
    chi2_central = np.sum(((y_mean - y_model) / y_err)**2)
    
    print(f"\n  Central fit results:")
    print(f"    c_X,0 (norm)  = {params_central[0]:.4f}")
    print(f"    c_X,1 (M_π²)  = {params_central[1]:.4f}")
    print(f"    c_X,2 (E)     = {params_central[2]:.4f}")
    print(f"    c_X,3 (E²)    = {params_central[3]:.4f}")
    print(f"    c_X,4 (a²)    = {params_central[4]:.4f}")
    print(f"    χ² = {chi2_central:.2f}")
    
    # Jackknife resampling
    params_jk = np.zeros((N_jk, 5))
    print(f"\n  Running {N_jk} jackknife fits...")
    
    n_failed = 0
    for i in range(N_jk):
        y_jk = data_dict['jk_samples'][i]
        
        def res_jk(params):
            y_model = evaluate_model(params, data_dict['E_K'], data_dict['M_pi'],
                                    data_dict['a'], Lambda, Delta_X)
            return (y_jk - y_model) * weights
        
        res_jk = least_squares(res_jk, params_central, method='lm',
                              ftol=1e-8, xtol=1e-8, max_nfev=5000, verbose=0)
        
        if res_jk.success:
            params_jk[i] = res_jk.x
        else:
            params_jk[i] = params_central
            n_failed += 1
    
    if n_failed > 0:
        print(f"  WARNING: {n_failed}/{N_jk} jackknife fits failed")
    
    params_mean = np.mean(params_jk, axis=0)
    params_err = np.sqrt((N_jk - 1) * np.mean((params_jk - params_mean)**2, axis=0))
    
    dof = len(y_mean) - 5
    p_value = chi2_dist.sf(chi2_central, dof)
    
    return {
        'params': params_central,
        'params_err': params_err,
        'params_jk': params_jk,
        'chi2': chi2_central,
        'dof': dof,
        'p_value': p_value,
        'cov': cov,
        'y_central': y_mean,
        'y_err': y_err,
        'Lambda': Lambda,
        'Delta_X': Delta_X
    }


def plot_results(FF, ensemble_data, fit_result, data_dict):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ensemble_colors = {
        "F1S": "#43A047",
        "M1": plt.cm.Blues(0.55),
        "M2": plt.cm.Blues(0.70),
        "M3": plt.cm.Blues(0.85),
        "C1": plt.cm.Reds(0.55),
        "C2": plt.cm.Reds(0.85),
    }
    
    # Plot data
    idx = 0
    for ens, data in ensemble_data.items():
        n_pts = len(data['q2'])
        x_vals = data['q2']
        y_vals = data_dict['y_central'][idx:idx+n_pts]
        y_err = fit_result['y_err'][idx:idx+n_pts]
        
        ax.errorbar(x_vals, y_vals, yerr=y_err, fmt=marker_styles[ens],
                    color=ensemble_colors[ens], capsize=4, markersize=8,
                    linewidth=1.5, label=f"{ens}", zorder=10)
        idx += n_pts
    
    # Plot fit curves with error bands
    Lambda = fit_result['Lambda']
    Delta_X = fit_result['Delta_X']
    N_jk = fit_result['params_jk'].shape[0]
    
    for ens, data in ensemble_data.items():
        E_K_min, E_K_max = data['E_K'].min(), data['E_K'].max()
        E_K_fine = np.linspace(E_K_min, E_K_max, 100)
        M_pi_ens = data['M_pi'][0]
        a_ens = data['a'][0]
        
        # Central fit
        y_fit = evaluate_model(fit_result['params'], E_K_fine,
                              np.full_like(E_K_fine, M_pi_ens),
                              np.full_like(E_K_fine, a_ens),
                              Lambda, Delta_X)
        
        # Jackknife error band
        y_jk_curves = np.zeros((N_jk, len(E_K_fine)))
        for i in range(N_jk):
            y_jk_curves[i] = evaluate_model(fit_result['params_jk'][i], E_K_fine,
                                           np.full_like(E_K_fine, M_pi_ens),
                                           np.full_like(E_K_fine, a_ens),
                                           Lambda, Delta_X)
        
        y_mean = np.mean(y_jk_curves, axis=0)
        y_std = np.sqrt((N_jk - 1) * np.mean((y_jk_curves - y_mean)**2, axis=0))
        
        # Convert E_K to q² for plotting
        MBs = data['MBs']
        q2_fine = (E_K_fine / MBs)**2
        
        # Plot central curve
        ax.plot(q2_fine, y_fit, '-', color=ensemble_colors[ens],
                linewidth=2.5, alpha=0.9, zorder=5)
        
        # Plot error band
        ax.fill_between(q2_fine, y_fit - y_std, y_fit + y_std,
                       color=ensemble_colors[ens], alpha=0.2, zorder=4)
    
    # Continuum limit with error band
    if show_continuum:
        E_K_cont = np.linspace(data_dict['E_K'].min(), data_dict['E_K'].max(), 200)
        
        # Central continuum curve
        y_cont = evaluate_model(fit_result['params'], E_K_cont,
                               np.full_like(E_K_cont, M_pi_phys),
                               np.zeros_like(E_K_cont),
                               Lambda, Delta_X)
        
        # Jackknife error band for continuum
        y_cont_jk = np.zeros((N_jk, len(E_K_cont)))
        for i in range(N_jk):
            y_cont_jk[i] = evaluate_model(fit_result['params_jk'][i], E_K_cont,
                                         np.full_like(E_K_cont, M_pi_phys),
                                         np.zeros_like(E_K_cont),
                                         Lambda, Delta_X)
        
        y_cont_mean = np.mean(y_cont_jk, axis=0)
        y_cont_std = np.sqrt((N_jk - 1) * np.mean((y_cont_jk - y_cont_mean)**2, axis=0))
        
        # Use average MBs for continuum curve
        MBs_avg = np.mean([data['MBs'] for data in ensemble_data.values()])
        q2_cont = (E_K_cont / MBs_avg)**2
        
        # Plot continuum curve and band
        ax.plot(q2_cont, y_cont, 'k--', linewidth=3,
                label='Continuum', zorder=3)
        ax.fill_between(q2_cont, y_cont - y_cont_std, y_cont + y_cont_std,
                       color='gray', alpha=0.3, zorder=2)
    
    ax.set_xlabel(r'$(E_{D_s^*}/M_{B_s})^2$', fontsize=20)
    ax.set_ylabel(f'{FF}', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, framealpha=0.9, loc='best', ncol=2)
    ax.tick_params(labelsize=14)
    ax.annotate(r'$\bf{preliminary}$', xy=(0.17, 0.03), xycoords='axes fraction',
             fontsize=20, color='grey', alpha=.7)
    
    plt.tight_layout()
    return fig


# ============================================================
# SAVE FIT RESULTS
# ============================================================

def save_fit_results(FF, fit_result, Lambda, Delta_X, filename=None):
    """Save fit results to CSV file."""
    if filename is None:
        filename = f"{FF}-Fit-Results.csv"
    
    # Create results dictionary
    results = {
        'FormFactor': [FF],
        'Lambda_GeV': [Lambda],
        'Delta_X_GeV': [Delta_X],
        'chi2': [fit_result['chi2']],
        'dof': [fit_result['dof']],
        'chi2_dof': [fit_result['chi2'] / fit_result['dof']],
        'p_value': [fit_result['p_value']],
        'c0_norm': [fit_result['params'][0]],
        'c0_norm_err': [fit_result['params_err'][0]],
        'c1_Mpi2': [fit_result['params'][1]],
        'c1_Mpi2_err': [fit_result['params_err'][1]],
        'c2_E': [fit_result['params'][2]],
        'c2_E_err': [fit_result['params_err'][2]],
        'c3_E2': [fit_result['params'][3]],
        'c3_E2_err': [fit_result['params_err'][3]],
        'c4_a2': [fit_result['params'][4]],
        'c4_a2_err': [fit_result['params_err'][4]],
    }
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False, float_format='%.6f')
    print(f"  Fit results saved to: {filename}")


# ============================================================
# MAIN
# ============================================================

def main():
    variables = load_renorm_factors()
    
    # Form-factor-specific pole parameters for Bs → Ds*
    pole_params = {
        "V": {"Lambda": 1.0, "Delta_X": 0.04845},
        "A0": {"Lambda": 1.0, "Delta_X": 0.26209},
        "A1": {"Lambda": 1.0, "Delta_X": 0.46174},
    }
    
    for FF in FF_list:
        print(f"\n{'='*60}\nAnalyzing {FF}\n{'='*60}")
        
        ensemble_data = build_jackknife_samples(FF, variables)
        if not ensemble_data:
            continue
        
        data_dict = create_combined_data(ensemble_data)
        
        Lambda_fit = pole_params[FF]["Lambda"]
        Delta_X_fit = pole_params[FF]["Delta_X"]
        
        print(f"\nPole parameters: Λ={Lambda_fit:.3f} GeV, Δ_X={Delta_X_fit:.3f} GeV")
        
        fit_result = fit_with_jackknife(data_dict, Lambda_fit, Delta_X_fit)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS FOR {FF}")
        print(f"{'='*60}")
        print(f"χ²/dof = {fit_result['chi2']/fit_result['dof']:.3f}")
        print(f"p-value = {fit_result['p_value']:.4f}")
        print(f"\nFitted parameters:")
        print(f"  c_X,0 (norm)  = {fit_result['params'][0]:.4f} ± {fit_result['params_err'][0]:.4f}")
        print(f"  c_X,1 (M_π²)  = {fit_result['params'][1]:.4f} ± {fit_result['params_err'][1]:.4f}")
        print(f"  c_X,2 (E)     = {fit_result['params'][2]:.4f} ± {fit_result['params_err'][2]:.4f}")
        print(f"  c_X,3 (E²)    = {fit_result['params'][3]:.4f} ± {fit_result['params_err'][3]:.4f}")
        print(f"  c_X,4 (a²)    = {fit_result['params'][4]:.4f} ± {fit_result['params_err'][4]:.4f}")
        
        # Save fit results to CSV
        save_fit_results(FF, fit_result, Lambda_fit, Delta_X_fit)
        
        fig = plot_results(FF, ensemble_data, fit_result, data_dict)
        outname = f"{FF}-Fit-NoChiralLog.png"
        fig.savefig(outname, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: {outname}")
        plt.close()


if __name__ == "__main__":
    main()