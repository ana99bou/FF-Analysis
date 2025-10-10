import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- config ----------
id_to_label = {"005": "C1", "010": "C2", "004": "M1", "006": "M2", "008": "M3", "002144": "F1S"}
ow_base = Path("../Results/Crosschecks/OW/Excited")
ab_base = Path("../Results/Crosschecks/AB")
id_str = "002144"
amc = 0.275
label = id_to_label[id_str]



# ---------- helpers ----------
_float_token_re = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")

def load_ow_file(path: Path) -> pd.DataFrame:
    """OW is whitespace: columns: index value uncertainty"""
    df = pd.read_csv(path, delim_whitespace=True)
    # ensure correct dtypes and expected columns
    df = df.rename(columns=str.strip)
    assert {"index", "value", "uncertainty"}.issubset(df.columns), f"OW file {path} missing columns"
    df["index"] = df["index"].astype(int)
    df = df.sort_values("index").reset_index(drop=True)
    return df[["index", "value", "uncertainty"]]

def load_ab_file(path: Path) -> pd.DataFrame:
    """
    Robust reader for AB:
    - Lines may be whitespace-separated, with an *index* in the first column.
    - Expected numeric layout per data row:
        idx  value  error  [pval ...]   -> use columns 1 and 2
      or value error pval               -> if no leading idx is present
    - Skips headers/comments.
    """
    rows = []
    idx_counter = 0
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # skip header-ish lines
            if any(h in line.lower() for h in ("value", "error", "pval")):
                continue
            nums = _float_token_re.findall(line)
            if len(nums) >= 4:
                # assume: idx, value, error, pval...
                try:
                    idx = int(float(nums[0]))
                except ValueError:
                    # fallback: treat first number as value (no leading index)
                    rows.append({"index": idx_counter, "value": float(nums[0]), "uncertainty": float(nums[1])})
                    idx_counter += 1
                else:
                    rows.append({"index": idx, "value": float(nums[1]), "uncertainty": float(nums[2])})
            elif len(nums) >= 2:
                # no explicit index; just  value error [pval]
                rows.append({"index": idx_counter, "value": float(nums[0]), "uncertainty": float(nums[1])})
                idx_counter += 1
            # else: ignore short/odd lines
    if not rows:
        raise RuntimeError(f"No numeric rows parsed from {path}")
    df = pd.DataFrame(rows)
    # If duplicate indices appear, keep the first occurrence (or you could average)
    df = df.drop_duplicates(subset=["index"]).sort_values("index").reset_index(drop=True)
    return df[["index", "value", "uncertainty"]]

# ---------- load ----------
fn_ow = ow_base / f"{id_str}-amc{amc:.3f}.txt"
# Try AB with and without .csv
candidates_ab = [
    ab_base / f"Crosscheck-excited-{label}-{amc:.3f}.csv",
    ab_base / f"Crosscheck-excited-{label}-{amc:.3f}",
]
fn_ab = next((p for p in candidates_ab if p.exists()), None)
if fn_ab is None:
    raise FileNotFoundError("Could not find AB file with or without .csv extension.")

df_ow = load_ow_file(fn_ow)
df_ab = load_ab_file(fn_ab)

# Align on index that exists in both (safest)
merged = pd.merge(
    df_ow.add_suffix("_ow"), df_ab.add_suffix("_ab"),
    left_on="index_ow", right_on="index_ab", how="inner"
)

# If their index columns are guaranteed identical (0..20), we can simplify:
# merged['index'] = merged['index_ow']
# Otherwise, use the left/right index to be explicit:
merged["index"] = merged["index_ow"].astype(int)

print(f"Loaded OW rows: {len(df_ow)}, AB rows: {len(df_ab)}, merged rows: {len(merged)}")
if len(merged) == 0:
    raise RuntimeError("No overlapping indices between OW and AB — check the AB index parsing.")

# ---------- plot ----------

import numpy as np

# --- define tick labels ---
xtick_labels = [
    r"$m_0^{B_s}$",
    r"$m_0^{D_s^*}(0)$", r"$m_0^{D_s^*}(1)$", r"$m_0^{D_s^*}(2)$",
    r"$m_0^{D_s^*}(3)$", r"$m_0^{D_s^*}(4)$", r"$m_0^{D_s^*}(5)$",
    r"$m_1^{B_s}$",
    r"$m_1^{D_s^*}(0)$", r"$m_1^{D_s^*}(1)$", r"$m_1^{D_s^*}(2)$",
    r"$m_1^{D_s^*}(3)$", r"$m_1^{D_s^*}(4)$", r"$m_1^{D_s^*}(5)$",
    r"$\delta m^{B_s}$",
    r"$\delta m^{D_s^*}(0)$", r"$\delta m^{D_s^*}(1)$", r"$\delta m^{D_s^*}(2)$",
    r"$\delta m^{D_s^*}(3)$", r"$\delta m^{D_s^*}(4)$", r"$\delta m^{D_s^*}(5)$"
]

save_dir = Path("../Results/Crosschecks")
save_dir.mkdir(parents=True, exist_ok=True)

# --- define subsets ---
plots = {
    "DsStar-masses": {
        "indices": [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13],
        "title": r"$m_0^{D_s^*}$ and $m_1^{D_s^*}$"
    },
    "Bs-masses": {
        "indices": [0, 7],
        "title": r"$m_0^{B_s}$ and $m_1^{B_s}$"
    },
    "Delta": {
        "indices": [14, 15, 16, 17, 18, 19, 20],
        "title": r"$\Delta m^{B_s}$ and $\Delta m^{D_s^*}$"
    },
}

# --- plotting loop ---
for name, cfg in plots.items():
    subset = merged[merged["index"].isin(cfg["indices"])].copy()
    subset = subset.sort_values("index")

    plt.figure(figsize=(8, 4))
    plt.errorbar(
        subset["index"], subset["value_ow"], yerr=subset["uncertainty_ow"],
        fmt="x", color="red", label="OW"
    )
    plt.errorbar(
        subset["index"], subset["value_ab"], yerr=subset["uncertainty_ab"],
        fmt="x", color="blue", label="AB"
    )

    # get tick labels corresponding to those indices
    ticks = subset["index"].to_numpy()
    labels = [xtick_labels[i] for i in ticks]

    plt.xticks(ticks=ticks, labels=labels, rotation=45, ha="right")
    plt.xlabel("Observable")
    plt.ylabel("Value")
    plt.title(f"{cfg['title']} — {label}, amc={amc:.3f}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # save file
    outfile = save_dir / f"Crosscheck-{label}-amc{amc:.3f}-{name}.png"
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅ Saved {outfile}")


'''
# labels for x-axis
xtick_labels = [
    r"$m_0^{B_s}$",
    r"$m_0^{D_s^*}(0)$", r"$m_0^{D_s^*}(1)$", r"$m_0^{D_s^*}(2)$",
    r"$m_0^{D_s^*}(3)$", r"$m_0^{D_s^*}(4)$", r"$m_0^{D_s^*}(5)$",
    r"$m_1^{B_s}$",
    r"$m_1^{D_s^*}(0)$", r"$m_1^{D_s^*}(1)$", r"$m_1^{D_s^*}(2)$",
    r"$m_1^{D_s^*}(3)$", r"$m_1^{D_s^*}(4)$", r"$m_1^{D_s^*}(5)$",
    r"$\delta m^{B_s}$",
    r"$\delta m^{D_s^*}(0)$", r"$\delta m^{D_s^*}(1)$", r"$\delta m^{D_s^*}(2)$",
    r"$\delta m^{D_s^*}(3)$", r"$\delta m^{D_s^*}(4)$", r"$\delta m^{D_s^*}(5)$"
]

# make plot
plt.figure(figsize=(11,5))
plt.errorbar(
    merged["index"], merged["value_ow"], yerr=merged["uncertainty_ow"],
    fmt="x", color="red", label="OW"
)
plt.errorbar(
    merged["index"], merged["value_ab"], yerr=merged["uncertainty_ab"],
    fmt="x", color="blue", label="AB"
)

plt.xticks(ticks=range(len(xtick_labels)), labels=xtick_labels, rotation=45, ha="right")
plt.xlabel("Observable")
plt.ylabel("Value")
plt.title(f"Crosscheck {id_str} ({label}), amc={amc:.3f}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# save to Results/Crosschecks with label+amc in filename
save_dir = Path("../Results/Crosschecks")
save_dir.mkdir(parents=True, exist_ok=True)
plot_file = save_dir / f"Crosscheck-2pt-{label}-amc{amc:.3f}.png"
plt.savefig(plot_file, dpi=300)
plt.close()

print(f"✅ Plot saved to {plot_file}")
'''