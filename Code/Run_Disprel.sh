#!/bin/bash

# Set ensemble and cmass
Ensemble="C2"
cmass_index=2

# Option 1: particle = Bs, nsq = 0
particle="Bs"
nsq=0
echo "Running with Ensemble=$Ensemble, particle=$particle, nsq=$nsq, cmass_index=$cmass_index"
python3 DispRel.py --ensemble "$Ensemble" --particle "$particle" --nsq "$nsq" --cmass_index "$cmass_index"

# Option 2: particle = Ds, nsq = 0 to 5
particle="Ds"
echo 'Running Ds'
python3 Disp-Rel.py --ensemble "$Ensemble" --particle "$particle" --nsq "$nsq" --cmass_index "$cmass_index"