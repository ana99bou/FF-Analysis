import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Function to read CSV files and return data
def read_csv_files(pattern, names, skiprows=1):
    files = sorted(glob.glob(pattern))
    print(files)
    data = []
    for file in files:
        df = pd.read_csv(file, sep='\s+', names=names, skiprows=skiprows)
        data.append(df)
    return data

# Read all Mass-Ds0.400-*.csv files
mass_data = read_csv_files('Mass-Ds0.400-*.csv', names=['Time', 'EffectiveMass', 'Error'])

# Read all Ds0.400Result-*.csv files
result_data = read_csv_files('Ds0.400Result-*.csv', names=['Time', 'EffectiveMass', 'Error', 'RegUp', 'RegLow'])

print(result_data)

# Plot the data
plt.xlabel('Time')
plt.ylabel(r'Effective Mass $D_s^*$ C1')
plt.ylim(1.0, 1.6)  # Restrict the y-axis to the range from 0.8 to 1.2

colors = ['g', 'b', 'orange', 'brown', 'red', 'cyan']
labels = [f'$n^2={i}$' for i in range(len(mass_data))]

for i, (mass_df, result_df) in enumerate(zip(mass_data, result_data)):
    plt.errorbar(mass_df['Time'], mass_df['EffectiveMass'], yerr=mass_df['Error'], ls='none', fmt='x', label=labels[i], color=colors[i])
    
    reg_low = int(result_df['RegLow'].iloc[0])
    reg_up = int(result_df['RegUp'].iloc[0])
    effective_mass = result_df['EffectiveMass'].iloc[0]
    
    plt.plot([reg_low, reg_up], [effective_mass, effective_mass], color=colors[i])

plt.legend(ncol=2)
plt.savefig('EffectiveMassPlot.pdf')
plt.show()