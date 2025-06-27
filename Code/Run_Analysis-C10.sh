#!/bin/bash

# Fixed inputs
ensemble="C2"
cmass_index=0

# Define form factors and their corresponding nsq values
declare -A nsq_values
nsq_values["V"]="1 2 3 4 5"
nsq_values["A0"]="1 2 3 4 5"
nsq_values["A1"]="0 1 2 4 5"

# Loop over form factors
for FF in "${!nsq_values[@]}"; do
  for nsq in ${nsq_values[$FF]}; do
    echo "Running with FF=$FF, nsq=$nsq, cmass_index=$cmass_index, ensemble=$ensemble"
    python3 Analysis.py "$FF" "$nsq" "$cmass_index" "$ensemble"
  done
done
