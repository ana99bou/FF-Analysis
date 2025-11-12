import glob
import re
import pandas as pd
import os
import numpy as np


######### Read Renormalization factors
variables = {}

for filename in glob.glob("../Data/Renorm/output-*.txt"):
    with open(filename, "r") as f:
        lines = f.readlines()
        header = lines[0].strip().split("\t")
        values = lines[1].strip().split("\t")

        # Extract label from filename
        label = filename.split("-")[1].split(".")[0]

        # === ZAll ===
        try:
            zall_index = header.index("ZAll")
            zall_val = re.sub(r"\([^)]*\)", "", values[zall_index].strip())
            if zall_val:
                variables[f"Zall_{label}"] = float(zall_val)
        except ValueError:
            print(f"No ZAll column found in {filename}")

        # === ZAccs ===
        zacc_indices = [i for i, col in enumerate(header) if col.strip() == "ZAcc"]
        for idx, zacc_index in enumerate(zacc_indices):
            val = values[zacc_index].strip()
            val = re.sub(r"\([^)]*\)", "", val)
            if val:
                try:
                    variables[f"Zacc_{label}_m{idx+1}"] = float(val)
                except ValueError:
                    print(f"Couldn't convert ZAcc value '{val}' in {filename}")

        # === ZVbb ===
        try:
            zvbb_index = header.index("ZVbb")
            zvbb_val = re.sub(r"\([^)]*\)", "", values[zvbb_index].strip())
            if zvbb_val:
                variables[f"Zvbb_{label}"] = float(zvbb_val)
        except ValueError:
            print(f"No ZVbb column found in {filename}")

# Print results sorted by variable name
for name, value in sorted(variables.items()):
    print(f"{name} = {value}")

with open("../Data/Renorm/rho.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # skip malformed lines

        label = parts[0]  # e.g. C, M, F

        # Use regex to extract numbers without parentheses
        rho_v0_match = re.search(r"rho_V0\s*=\s*([\d.]+)", line)
        rho_vi_match = re.search(r"rho_Vi\s*=\s*([\d.]+)", line)

        if rho_v0_match:
            variables[f"rho_V0_{label}"] = float(rho_v0_match.group(1))
        if rho_vi_match:
            variables[f"rho_Vi_{label}"] = float(rho_vi_match.group(1))

# Print all variables
for name, value in sorted(variables.items()):
    print(f"{name} = {value}")

####### Read Results
import pandas as pd
import os
import matplotlib.pyplot as plt

inv_lat_sp = {
    "F1S": 2.785,  
    "M1": 2.3833,   
    "M2": 2.3833,   
    "M3": 2.3833,   
    "C1": 1.7848,   
    "C2": 1.7848    
}

ens_masses = {
    "F1S": ["0.259", "0.275"],
    "M1": ["0.280", "0.340"],
    "M2": ["0.280", "0.340"],
    "M3": ["0.280", "0.340"],
    "C1": ["0.300", "0.350", "0.400"],
    "C2": ["0.300", "0.350", "0.400"]
}

results_calc = {}

marker_styles = {
    "C1": 'x',
    "C2": 'o',
    "M1": '^',
    "M2": 's',
    "M3": 'D',
    "F1S": 'v',
}

# Base colormaps per prefix for similar colors in each ensemble family
base_colors = {
    "C": plt.cm.Reds,
    "M": plt.cm.Blues}
#bright_pinks = ["#e91e63", "#f06292", "#f48fb1"] 
bright_pinks = ["#43A047", "#66BB6A", "#9CCC65"]

def get_rho_prefix(ens):
    if ens.startswith("C"):
        return "C"
    elif ens.startswith("M"):
        return "M"
    elif ens.startswith("F"):
        return "F"
    else:
        raise ValueError(f"Unknown ensemble prefix for {ens}")

plt.figure(figsize=(12, 8))

for ens, masses in ens_masses.items():
    results_calc[ens] = {}
    rho_prefix = get_rho_prefix(ens)
    rho_val = variables.get(f"rho_Vi_{rho_prefix}", None)
    if rho_val is None:
        print(f"Missing rho_Vi_{rho_prefix} in variables dict, skipping {ens}")
        continue
    

    prefix = ens[0]  # e.g. 'C', 'M', 'F'
    n_masses = len(masses)

    if prefix == "F":
        colors = bright_pinks[:n_masses]
    else:
        cmap = base_colors.get(prefix, plt.cm.Greys)
        colors = [cmap(0.5 + 0.35 * i / max(n_masses - 1, 1)) for i in range(n_masses)]

    marker = marker_styles.get(ens, 'o')  # ensemble marker, fallback to 'o'

    for i, m in enumerate(masses):
        m_index = i + 1
        zacc_key = f"Zacc_{ens}_m{m_index}"
        zvbb_key = f"Zvbb_{ens}"
        
        zacc_val = variables.get(zacc_key, None)
        zvbb_val = variables.get(zvbb_key, None)
        
        if zacc_val is None or zvbb_val is None:
            print(f"Missing Zacc or Zvbb for {ens} {m}")
            continue
        
        prefactor = rho_val * np.sqrt(zacc_val * zvbb_val)
        
        #if ens == 'F1S':
        #    filepath = f"../Results/Crosschecks/AB/Crosscheck-{ens}-{m}.csv"
        #else:
        filepath = f"../Results/Crosschecks/AB/Crosscheck-excited-{ens}-{m}-V-3pt.csv"    
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        try:
            df = pd.read_csv(filepath, sep="\t")
            
            if len(df) < 12:
                print(f"File {filepath} has less than 12 rows, skipping")
                continue

            val0 = df.iloc[0]["Value"]
            val1 = df.iloc[1]["Value"]
            
            vals_list = []
            for x in range(1, 7):
                valx = df.iloc[x]["Value"]
                calc = inv_lat_sp[ens]**2 * (val0**2 + val1**2 - 2 * val0 * valx)
                vals_list.append(calc)
            
            x_vals = vals_list[1:]
            y_vals_raw = df.iloc[7:12]["Value"].tolist()
            y_errs_raw = df.iloc[7:12]["Error"].tolist()
            
            #print(x_vals)

            if len(x_vals) != len(y_vals_raw):
                print(f"Length mismatch for {ens} {m}")
                continue
            
            y_vals = [prefactor * y for y in y_vals_raw]
            y_errs = [prefactor * err for err in y_errs_raw]
            
            results_calc[ens][m] = (x_vals, y_vals)
            
            plt.errorbar(
                x_vals, y_vals, yerr=y_errs,
                fmt=marker,
                color=colors[i],
                capsize=3,
                label=f"{ens} @ {m}"
            )
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

plt.legend()
plt.xlabel(r"$q^2 [GeV]$",fontsize=20)  # z. B. "Energy [GeV²]"
plt.ylabel(r"V",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.annotate(r'$\bf{preliminary}$', xy=(0.17, 0.03), xycoords='axes fraction',
             fontsize=20, color='grey', alpha=.7)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('V-FPlot.png',dpi=200,bbox_inches='tight')


# === Neuer Plot für A_0 ===
plt.figure(figsize=(12, 8))

for ens, masses in ens_masses.items():
    rho_prefix = get_rho_prefix(ens)
    rho_val = variables.get(f"rho_Vi_{rho_prefix}", None)
    if rho_val is None:
        continue

    prefix = ens[0]
    n_masses = len(masses)

    if prefix == "F":
        colors = bright_pinks[:n_masses]
    else:
        cmap = base_colors.get(prefix, plt.cm.Greys)
        colors = [cmap(0.5 + 0.35 * i / max(n_masses - 1, 1)) for i in range(n_masses)]

    marker = marker_styles.get(ens, 'o')

    for i, m in enumerate(masses):
        m_index = i + 1
        zacc_key = f"Zacc_{ens}_m{m_index}"
        zvbb_key = f"Zvbb_{ens}"

        zacc_val = variables.get(zacc_key, None)
        zvbb_val = variables.get(zvbb_key, None)

        if zacc_val is None or zvbb_val is None:
            continue

        prefactor = rho_val * np.sqrt(zacc_val * zvbb_val)

        filepath = f"../Results/Crosschecks/AB/Crosscheck-excited-{ens}-{m}-A0-3pt.csv"
        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_csv(filepath, sep="\t")

            if len(df) < 12:
                print(f"File {filepath} too short for A₀ values")
                continue

            val0 = df.iloc[0]["Value"]
            val1 = df.iloc[1]["Value"]

            vals_list = []
            for x in range(1, 7):
                valx = df.iloc[x]["Value"]
                calc = inv_lat_sp[ens]**2 * (val0**2 + val1**2 - 2 * val0 * valx)
                vals_list.append(calc)

            x_vals = vals_list[1:]
            y_vals_raw = df.iloc[7:12]["Value"].tolist()
            y_errs_raw = df.iloc[7:12]["Error"].tolist()

            if len(x_vals) != len(y_vals_raw):
                print(f"Length mismatch for A₀ {ens} {m}")
                continue

            y_vals = [prefactor * y for y in y_vals_raw]
            y_errs = [prefactor * err for err in y_errs_raw]

            print('here:')
            print(x_vals)
            print(y_vals)
            print(y_errs)

            plt.errorbar(
                x_vals, y_vals, yerr=y_errs,
                fmt=marker,
                color=colors[i],
                capsize=3,
                label=f"{ens} @ {m}"
            )

        except Exception as e:
            print(f"Error processing {filepath} for A₀: {e}")

plt.legend()
plt.xlabel(r"$q^2\ [\mathrm{GeV}^2]$")
plt.ylabel(r"$A_0$")
plt.grid(True)
plt.tight_layout()
plt.savefig('A0-FPlot.pdf')


# === Neuer Plot für A_1 ===
plt.figure(figsize=(12, 8))

for ens, masses in ens_masses.items():
    rho_prefix = get_rho_prefix(ens)
    rho_val = variables.get(f"rho_Vi_{rho_prefix}", None)
    if rho_val is None:
        continue

    prefix = ens[0]
    n_masses = len(masses)

    if prefix == "F":
        colors = bright_pinks[:n_masses]
    else:
        cmap = base_colors.get(prefix, plt.cm.Greys)
        colors = [cmap(0.5 + 0.35 * i / max(n_masses - 1, 1)) for i in range(n_masses)]

    marker = marker_styles.get(ens, 'o')

    for i, m in enumerate(masses):
        m_index = i + 1
        zacc_key = f"Zacc_{ens}_m{m_index}"
        zvbb_key = f"Zvbb_{ens}"

        zacc_val = variables.get(zacc_key, None)
        zvbb_val = variables.get(zvbb_key, None)

        if zacc_val is None or zvbb_val is None:
            continue

        prefactor = rho_val * np.sqrt(zacc_val * zvbb_val)

        filepath = f"../Results/Crosschecks/AB/Crosscheck-excited-{ens}-{m}-A1-3pt.csv"
        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_csv(filepath, sep="\t")

            if len(df) < 12:
                print(f"File {filepath} too short for A₁ values")
                continue

            val0 = df.iloc[0]["Value"]
            val1 = df.iloc[1]["Value"]

            vals_list = []
            for x in range(1, 7):
                valx = df.iloc[x]["Value"]
                calc = inv_lat_sp[ens]**2 * (val0**2 + val1**2 - 2 * val0 * valx)
                vals_list.append(calc)

            # x-Werte für A1: [0], [1], [2], [4], [5]
            x_vals = [vals_list[i] for i in [0, 1, 2, 4, 5]]
            y_vals_raw = df.iloc[7:12]["Value"].tolist()
            y_errs_raw = df.iloc[7:12]["Error"].tolist()

            if len(x_vals) != len(y_vals_raw):
                print(f"Length mismatch for A₁ {ens} {m}")
                continue

            y_vals = [prefactor * y for y in y_vals_raw]
            y_errs = [prefactor * err for err in y_errs_raw]

            plt.errorbar(
                x_vals, y_vals, yerr=y_errs,
                fmt=marker,
                color=colors[i],
                capsize=3,
                label=f"{ens} @ {m}"
            )

        except Exception as e:
            print(f"Error processing {filepath} for A₁: {e}")

plt.legend()
plt.xlabel(r"$q^2\ [\mathrm{GeV}^2]$")
plt.ylabel(r"$A_1$")
plt.grid(True)
plt.tight_layout()
plt.savefig('A1-FPlot.pdf')
