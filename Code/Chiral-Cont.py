import csv
from pathlib import Path
import numpy as np

# Define the FF you want to read (currently fixed to "V")
FF = "V"

# Your ensemble → cmass mapping
ens_dict = {
    "F1S": [0.259, 0.275],
    "M1":  [0.280, 0.340],
    "M2":  [0.280, 0.340],
    "M3":  [0.280, 0.340],
    "C1":  [0.300, 0.350, 0.400],
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

    print("Total number of super-JK blocks:", len(super_jk))
    example_j = 0
    print("Number of ensemble–cmass pairs in each block:",
      len(super_jk[example_j]))
    example_j = 0
    print("\nq² lengths per (Ens, cmass):")
    for key, vec in super_jk[example_j].items():
        print(f"{key}: length = {len(vec)}")
        print(vec)

    print("\nSummary of shapes for all super-JK blocks:")
    for j, block in enumerate(super_jk):
        shapes = {key: len(vec) for key, vec in block.items()}
        print(f"Block {j}: {shapes}")
        if j == 2:  
            break

    
    print("\nConsistency check across all blocks:")

    first_keys = set(super_jk[0].keys())
    first_shapes = {k: len(super_jk[0][k]) for k in first_keys}

    for j, block in enumerate(super_jk):
        if set(block.keys()) != first_keys:
           print("❌ Key mismatch in block", j)
           break

        shapes = {k: len(block[k]) for k in block}
        if shapes != first_shapes:
            print("❌ Shape mismatch in block", j)
            print("Expected:", first_shapes)
            print("Found   :", shapes)
            break
    else:
        print("✅ All blocks consistent.")



    return super_jk

print(np.array(build_super_jackknife(results)))