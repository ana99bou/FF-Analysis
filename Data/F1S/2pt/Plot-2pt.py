import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cmass = '0.248'
num_files = 6

effective_mass = []
errors = []

for i in range(num_files):
    df = pd.read_csv(f'./Ds{cmass}Result-{i}.csv', sep='\s')
    effective_mass.append(df['EffectiveMass'][0])
    errors.append(df['Error'][0])

x_values = list(range(num_files))


def calculate_value(a, md, p, L):
    term1 = np.sinh(md / 2) ** 2
    term2 = np.sin(p[0] * np.pi / L) ** 2
    term3 = np.sin(p[1] * np.pi / L) ** 2
    term4 = np.sin(p[2] * np.pi / L) ** 2
    result = 2 * np.arcsinh(np.sqrt(term1 + term2 + term3 + term4))
    return result

disprel=[]
vecs=[[0,0,0],[1,0,0],[1,1,0],[1,1,1],[2,0,0],[2,1,0]]

for i in range(num_files):
    disprel.append(calculate_value(1/2.785,effective_mass[0],vecs[i],48))
print(disprel)

plt.errorbar(x_values, effective_mass, yerr=errors, fmt='o', capsize=5, label='Effective Mass')
plt.scatter(x_values, disprel, label='Dispersion Relation', color='red')
plt.xlabel(r'$n^2$')
plt.ylabel(r'Effective Eenrgy $D_s^*$ F1S')
#plt.title('Effective Mass vs File Index')
plt.legend()
#plt.grid(True)
plt.savefig('Disprel.pdf', bbox_inches='tight')
plt.show()