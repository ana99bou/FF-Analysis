import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Ensemble as Ens
import argparse

'''
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
'''

Ensemble = "F1S"
particle = "Ds"
nsq=0
cmass_index = 1


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
disperr=[errors[0] for _ in range(num_files)]
'''
plt.errorbar(x_values, effective_mass, yerr=errors, fmt='o', capsize=5,color='green', label='Fit')
plt.errorbar(x_values, disprel,yerr=disperr, label='Disp. Relation', color='crimson',fmt='^')
plt.xlabel(r'$n^2$', fontsize=16)
plt.ylabel(r'$E_{eff}$',fontsize=16)
#plt.title('Effective Mass vs File Index')
plt.legend()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.annotate(r'$\bf{preliminary}$', xy=(0.17, 0.03), xycoords='axes fraction',
             fontsize=15, color='grey', alpha=.7)
plt.legend(fontsize=14)
#plt.grid(True)
plt.savefig('../Results/DispRel/Disprel-{}-{}-{}.pdf'.format(Ensemble,cmass,particle), bbox_inches='tight', transparent=True)
'''

# Compute physical squared momenta (p_k^2 in GeV^2)
p_squared = []
pk=[]
a = 1 / inv
for vec in vecs:
    p2 = 0
    for i in range(3):
        p2 += (2*np.pi / (a* L))**2 * (vec[i]**2)
    p_squared.append(p2)

#Faktore vier

# Normalize the dispersion relation values
normalized_disp = [(inv**2*effective_mass[i]**2 / (inv**2*disprel[0]**2+p_squared[i])) for i in  range(len(disprel))]
tmp_disp=[((np.sqrt(disprel[0]**2+1/inv**2*p_squared[i]))) for i in  range(len(disprel))]
tmp_disp_err=[((1/np.sqrt(disprel[0]**2+1/inv**2*p_squared[i])*errors[0])) for i in  range(len(disprel))]
print(disprel)
print(disprel[0])
print(p_squared)
print(tmp_disp)
print(normalized_disp)


# Dashed lines: 1 ± a * p^2 / 4
pk2 = np.array(p_squared)
upper_line = 1 + a * pk2 / 8
lower_line = 1 - a * pk2 / 8


# Normiertes Verhältnis mit der kontinuierlichen Dispersionsrelation im Nenner
normalized_cont = [(effective_mass[i]**2 / (disprel[i]**2)) for i in range(len(disprel))]
cont_errors = [2 * effective_mass[i] * errors[i] / disprel[i]**2 for i in range(len(disprel))]




# Plotting the dispersion ratio plot
plt.figure()
plt.errorbar(p_squared, normalized_disp, yerr=[2 * val * err / disprel[0] for val, err in zip(disprel, errors)],
             fmt='s', color='navy', label='cont. disp. rel.')
plt.errorbar(p_squared, normalized_cont, yerr=cont_errors,
             fmt='^', color='crimson', capsize=4, label='lat. disp. rel.')

plt.plot(pk2, upper_line, linestyle='--', color='navy', alpha=0.7, label=r'$+ap_k^2/4$')
plt.plot(pk2, lower_line, linestyle='--', color='navy', alpha=0.7)

plt.axhline(1.0, color='black', linewidth=0.8)

plt.xlabel(r'$k^2$ [GeV$^2$]',fontsize=16)
plt.ylabel(r'$E^2 / (k^2 + M^2)$',fontsize=16)
plt.legend()
plt.tight_layout()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.annotate(r'$\bf{preliminary}$', xy=(0.17, 0.03), xycoords='axes fraction',
             fontsize=15, color='grey', alpha=.7)
plt.legend(fontsize=14)
plt.ylim(0.85, 1.15)
plt.savefig('../Results/DispRel/Ratio-Disprel-{}-{}-{}.pdf'.format(Ensemble, cmass, particle), bbox_inches='tight',transparent=True)


#######################

plt.figure()
plt.errorbar(x_values, effective_mass, yerr=errors, fmt='o', capsize=5,color='green', label='Fit')
plt.errorbar(x_values, disprel,yerr=disperr, label='Latt. Disp. Relation', color='crimson',fmt='^')
plt.errorbar(x_values, tmp_disp,yerr=tmp_disp_err, label='Cont. Disp. Relation', color='navy',fmt='s')
plt.xlabel(r'$n^2$', fontsize=16)
plt.ylabel(r'$E_{eff}$',fontsize=16)
#plt.title('Effective Mass vs File Index')
plt.legend()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.annotate(r'$\bf{preliminary}$', xy=(0.17, 0.03), xycoords='axes fraction',
             fontsize=15, color='grey', alpha=.7)
plt.legend(fontsize=14)
#plt.grid(True)
plt.savefig('../Results/DispRel/Disprel-{}-{}-{}.pdf'.format(Ensemble,cmass,particle), bbox_inches='tight', transparent=True)
