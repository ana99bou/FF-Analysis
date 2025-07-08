import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Ensemble as Ens
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ensemble', type=str, required=True)
parser.add_argument('--particle', type=str, required=True)
parser.add_argument('--nsq', type=int, required=True)
parser.add_argument('--cmass_index', type=int, required=True)
args = parser.parse_args()

# Use the parsed arguments
Ensemble = args.ensemble
particle = args.particle
nsq = args.nsq
cmass_index = args.cmass_index

cmass=Ens.getCmass(Ensemble)[cmass_index]
inv=Ens.getInvSpac(Ensemble)
nconf,dt,ts,L= Ens.getEns(Ensemble)


#if particle == 'Bs':
#    num_files = 1
#else:
num_files = 6

effective_mass = []
errors = []

for i in range(num_files):
    df = pd.read_csv(f'../Data/{Ensemble}/2pt/Ds{cmass}Result-{i}.csv', sep='\s')
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
    disprel.append(calculate_value(1/inv,effective_mass[0],vecs[i],L))
print(disprel)

plt.errorbar(x_values, effective_mass, yerr=errors, fmt='o', capsize=5, label='Effective Mass')
plt.scatter(x_values, disprel, label='Dispersion Relation', color='red')
plt.xlabel(r'$n^2$')
plt.ylabel(r'Effective Eenrgy $D_s^*$ F1S')
#plt.title('Effective Mass vs File Index')
plt.legend()
#plt.grid(True)
plt.savefig('../Results/DispRel/Disprel-{}-{}-{}.pdf'.format(Ensemble,cmass,particle), bbox_inches='tight')
plt.show()




        
