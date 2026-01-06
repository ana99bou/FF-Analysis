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

# Two ensemble sets to compare
ensemble_sets = {
    "With M1/M2": ["F1S", "M1", "M2", "M3", "C1", "C2"],
    "Without M1/M2": ["F1S", "M3", "C1", "C2"]
}

FF_list = ["V", "A0", "A1"]

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
    d["DeltaMpi_phys"] = d["DeltaMpi"] * a_inv
    d["MpiS_phys"] = d["MpiS"] * a_inv
    d["a_phys"] = 1.0 / a_inv

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
# FIT FUNCTION
# ============================================================

def fit_function(params, E_K, M_pi_sim, a, Lambda, Delta_X):
    c_X0, c_X1, c_X2, c_X3, c_X4 = params
    pole_factor = Lambda / (E_K + Delta_X)
    DeltaM_pi_sq = M_pi_sim**2 - M_pi_phys**2
    
    expansion = (
        c_X0 +
        c_X1 * (DeltaM_pi_sq / Lambda**2) +
        c_X2 * (E_K / Lambda) +
        c_X3 * (E_K / Lambda)**2 +
        c_X4 * (a * Lambda)**2
    )
    
    return pole_factor * expansion


def evaluate_model(params, E_K_arr, M_pi_arr, a_arr, Lambda, Delta_X):
    return np.array([fit_function(params, E, M, a, Lambda, Delta_X)
                     for E, M, a in zip(E_K_arr, M_pi_arr, a_arr)])


# ============================================================
# DATA LOADING
# ============================================================

def build_jackknife_samples(FF, variables, active_ensembles):
    """Build jackknife samples for specified ensembles."""
    nsq_phys = nsq_list_for_FF(FF)
    ensemble_data = {}
    
    for ens in active_ensembles:
        rho_prefix = get_rho_prefix(ens)
        rho = variables.get(f"rho_Vi_{rho_prefix}")
        zacc = variables.get(f"Zacc_{ens}_m1")
        zvbb = variables.get(f"Zvbb_{ens}")
        
        if (rho is None) or (zacc is None) or (zvbb is None):
            continue
            
        prefactor = rho * np.sqrt(zacc * zvbb)
        
        m_choice = choose_m_for_ens(ens)
        path_bs = f"../Results/Crosschecks/AB/Crosscheck-excited-{ens}-{m_choice}-V-3pt.csv"
        if not os.path.exists(path_bs):
            continue
            
        MBs_latt = pd.read_csv(path_bs, sep="\t").iloc[0]["Value"]
        a_inv = inv_lat_sp[ens]
        MBs_phys = MBs_latt * a_inv
        
        path_jk = f"../Results/{ens}/Charm/PhysResults-JK-{FF}.csv"
        if not os.path.exists(path_jk):
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
            
            aE = Ds_mass_phys[ens][nsq]
            E_phys = aE * a_inv
            E_K_vals.append(E_phys)
            q2_vals.append((E_phys / MBs_phys)**2)
            M_pi_vals.append(fit_inputs[ens]["MpiS_phys"])
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
    
    E_K_all = np.zeros(n_total)
    M_pi_all = np.zeros(n_total)
    a_all = np.zeros(n_total)
    q2_all = np.zeros(n_total)
    ens_labels = []
    
    jk_combined = np.zeros((N_jk_total, n_total))
    y_central = np.zeros(n_total)
    
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
    
    weights = 1.0 / y_err
    
    def residuals(params):
        y_model = evaluate_model(params, data_dict['E_K'], data_dict['M_pi'],
                                 data_dict['a'], Lambda, Delta_X)
        return (y_mean - y_model) * weights
    
    E_mean = np.mean(data_dict['E_K'])
    y_mean_val = np.mean(y_mean)
    c_X0_guess = y_mean_val * (E_mean + Delta_X) / Lambda
    
    p0 = np.array([c_X0_guess, 0.0, 0.0, 0.0, 0.0])
    
    result = least_squares(residuals, p0, method='lm', ftol=1e-10, xtol=1e-10,
                          max_nfev=10000, verbose=0)
    
    params_central = result.x
    y_model = evaluate_model(params_central, data_dict['E_K'], data_dict['M_pi'],
                            data_dict['a'], Lambda, Delta_X)
    chi2_central = np.sum(((y_mean - y_model) / y_err)**2)
    
    # Jackknife resampling
    params_jk = np.zeros((N_jk, 5))
    
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
        'Lambda': Lambda,
        'Delta_X': Delta_X
    }


# ============================================================
# COMPARISON PLOTTING
# ============================================================

def plot_continuum_comparison(FF, results_dict):
    """
    Plot continuum extrapolations from different ensemble sets.
    
    Args:
        FF: Form factor name
        results_dict: Dict with keys from ensemble_sets, values are fit_results
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = {
        "With M1/M2": "#1f77b4",      # Blue
        "Without M1/M2": "#d62728"    # Red
    }
    
    linestyles = {
        "With M1/M2": "-",
        "Without M1/M2": "--"
    }
    
    # Determine E_K range for plotting (use widest range from all sets)
    E_K_min = min(results_dict[key]['E_K_range'][0] for key in results_dict)
    E_K_max = max(results_dict[key]['E_K_range'][1] for key in results_dict)
    
    # Average MBs for q² conversion
    MBs_avg = 5.367  # GeV, approximate Bs mass
    
    E_K_cont = np.linspace(E_K_min, E_K_max, 200)
    q2_cont = (E_K_cont / MBs_avg)**2
    
    # Plot each continuum extrapolation
    for set_name, fit_result in results_dict.items():
        Lambda = fit_result['Lambda']
        Delta_X = fit_result['Delta_X']
        N_jk = fit_result['params_jk'].shape[0]
        
        # Central continuum curve
        y_cont = evaluate_model(fit_result['params'], E_K_cont,
                               np.full_like(E_K_cont, M_pi_phys),
                               np.zeros_like(E_K_cont),
                               Lambda, Delta_X)
        
        # Jackknife error band
        y_cont_jk = np.zeros((N_jk, len(E_K_cont)))
        for i in range(N_jk):
            y_cont_jk[i] = evaluate_model(fit_result['params_jk'][i], E_K_cont,
                                         np.full_like(E_K_cont, M_pi_phys),
                                         np.zeros_like(E_K_cont),
                                         Lambda, Delta_X)
        
        y_cont_mean = np.mean(y_cont_jk, axis=0)
        y_cont_std = np.sqrt((N_jk - 1) * np.mean((y_cont_jk - y_cont_mean)**2, axis=0))
        
        # Plot
        label = f"{set_name} (χ²/dof={fit_result['chi2']/fit_result['dof']:.2f}, p={fit_result['p_value']:.3f})"
        ax.plot(q2_cont, y_cont, linestyle=linestyles[set_name], 
                color=colors[set_name], linewidth=3, zorder=5)
        ax.fill_between(q2_cont, y_cont - y_cont_std, y_cont + y_cont_std,
                       color=colors[set_name], alpha=0.25, zorder=3)
    
    ax.set_xlabel(r'$(E_{D_s^*}/M_{B_s})^2$', fontsize=20)
    ax.set_ylabel(f'{FF}', fontsize=20)
    #ax.set_title(f'{FF} Form Factor: Continuum Extrapolation Comparison', 
    #             fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, framealpha=0.95, loc='best')
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    variables = load_renorm_factors()
    
    pole_params = {
        "V": {"Lambda": 1.0, "Delta_X": 0.04845},
        "A0": {"Lambda": 1.0, "Delta_X": 0.26209},
        "A1": {"Lambda": 1.0, "Delta_X": 0.46174},
    }
    
    # Store results for comparison
    all_results = {}
    
    for FF in FF_list:
        print(f"\n{'='*70}")
        print(f"FORM FACTOR: {FF}")
        print('='*70)
        
        all_results[FF] = {}
        
        Lambda_fit = pole_params[FF]["Lambda"]
        Delta_X_fit = pole_params[FF]["Delta_X"]
        
        # Fit with each ensemble set
        for set_name, active_ensembles in ensemble_sets.items():
            print(f"\n--- {set_name} ---")
            print(f"Ensembles: {', '.join(active_ensembles)}")
            
            ensemble_data = build_jackknife_samples(FF, variables, active_ensembles)
            if not ensemble_data:
                print(f"  No data available")
                continue
            
            data_dict = create_combined_data(ensemble_data)
            
            print(f"  Points: {len(data_dict['y_central'])}")
            print(f"  Λ={Lambda_fit:.3f} GeV, Δ_X={Delta_X_fit:.3f} GeV")
            
            fit_result = fit_with_jackknife(data_dict, Lambda_fit, Delta_X_fit)
            
            print(f"  χ²/dof = {fit_result['chi2']/fit_result['dof']:.3f}")
            print(f"  p-value = {fit_result['p_value']:.4f}")
            print(f"  c0 = {fit_result['params'][0]:.4f} ± {fit_result['params_err'][0]:.4f}")
            
            # Store E_K range for plotting
            fit_result['E_K_range'] = (data_dict['E_K'].min(), data_dict['E_K'].max())
            
            all_results[FF][set_name] = fit_result
        
        # Create comparison plot
        if len(all_results[FF]) == 2:
            fig = plot_continuum_comparison(FF, all_results[FF])
            outname = f"{FF}-Continuum-Comparison.png"
            fig.savefig(outname, dpi=300, bbox_inches='tight')
            print(f"\n  Comparison plot saved: {outname}")
            plt.close()
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Continuum Extrapolation Comparison")
    print('='*70)
    
    for FF in FF_list:
        print(f"\n{FF}:")
        for set_name in ensemble_sets.keys():
            if set_name in all_results[FF]:
                res = all_results[FF][set_name]
                c0 = res['params'][0]
                c0_err = res['params_err'][0]
                chi2_dof = res['chi2'] / res['dof']
                print(f"  {set_name:20s}: c0 = {c0:.4f} ± {c0_err:.4f}  (χ²/dof = {chi2_dof:.2f})")


if __name__ == "__main__":
    main()
