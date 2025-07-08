#!/bin/bash

# Set ensemble and cmass
Ensemble="M1"
cmass_index=0

# Option 1: particle = Bs, nsq = 0
particle="Bs"
nsq=0
echo "Running with Ensemble=$Ensemble, particle=$particle, nsq=$nsq, cmass_index=$cmass_index"
python3 Effmass.py --ensemble "$Ensemble" --particle "$particle" --nsq "$nsq" --cmass_index "$cmass_index"

# Option 2: particle = Ds, nsq = 0 to 5
particle="Ds"
for nsq in {0..5}; do
    echo "Running with Ensemble=$Ensemble, particle=$particle, nsq=$nsq, cmass_index=$cmass_index"
    python3 Effmass.py --ensemble "$Ensemble" --particle "$particle" --nsq "$nsq" --cmass_index "$cmass_index"
done
python3 Disp-Rel.py --ensemble "$Ensemble" --particle "$particle" --nsq "$nsq" --cmass_index "$cmass_index"