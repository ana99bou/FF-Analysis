#!/bin/bash
# Usage: ./run_all_FF.sh <cmass_index> <ensemble> <use_disp> <frozen_analysis>
# Example: ./run_all_FF.sh 2 cA211a.40.24 1 0

# --- Fixed FF values ---
FF_list=("V" "A0")


cmass_index=$1
ensemble=$2
use_disp=$3
frozen_analysis=$4

# --- Check for correct usage ---
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <cmass_index> <ensemble> <use_disp> <frozen_analysis>"
    exit 1
fi

# --- Loop over all form factors ---
for FF in "${FF_list[@]}"; do
    echo "Running for FF=$FF, cmass_index=$cmass_index, ensemble=$ensemble, use_disp=$use_disp, frozen_analysis=$frozen_analysis"
    python Fit-Excited-Combined.py "$FF" dummy "$cmass_index" "$ensemble" "$use_disp" "$frozen_analysis"
done
