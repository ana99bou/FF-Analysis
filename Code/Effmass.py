### Current error. mass has NaN-> cosh for vaslues lower than one

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cmath
import pandas as pd
from scipy.optimize import minimize
import sys 
import os
import scipy
import Ensemble as Ens
import argparse


def pvalue(chi2, dof):
    r"""Compute the $p$-value corresponding to a $\chi^2$ with `dof` degrees
    of freedom."""
    return 1 - scipy.stats.chi2.cdf(chi2, dof)


def exp_val(data):
    return sum(data)/len(data)

def jack(x,j):
    r=0
    for i in range(len(x)):
        if i!=j:
            r=r+x[i]
    return 1/(len(x)-1)*r        


def var(data):
    res=0
    for i in range(len(data)):
        res=res+(jack(data,i)-exp_val(data))**2
    return np.sqrt((len(data)-1)/len(data)*res)

def extract(lst,number):
    return [item[number] for item in lst]

'''
def build_covmat_from_jackblock(mirtr, mass_block, reg_low, reg_up, exclude_idx):
    N = len(mirtr[0])  # configs
    T = reg_up - reg_low
    covmat = np.zeros((T, T))
    for t1 in range(T):
        for t2 in range(T):
            sum_term = 0
            for i in range(N):
                if i == exclude_idx:
                    continue
                m1 = np.arccosh((jack(mirtr[t1 + reg_low], i) + jack(mirtr[t1 + 2 + reg_low], i)) / (2 * jack(mirtr[t1 + 1 + reg_low], i)))
                m2 = np.arccosh((jack(mirtr[t2 + reg_low], i) + jack(mirtr[t2 + 2 + reg_low], i)) / (2 * jack(mirtr[t2 + 1 + reg_low], i)))
                sum_term += (m1 - mass_block[t1]) * (m2 - mass_block[t2])
            covmat[t1, t2] = (N - 1) / (N - 2) * sum_term / (N - 1)
    return np.linalg.inv(covmat)
'''
def build_covmat_from_jackblock(mirtr, reg_low, reg_up, exclude_idx):
    N = len(mirtr[0])  # configs
    T = reg_up - reg_low
    njack = N - 1

    # 1. Effektive Massen aller Jackknife-Blöcke (ohne exclude_idx)
    mass_all = np.zeros((njack, T))
    idx = 0
    for i in range(N):
        if i == exclude_idx:
            continue
        for t in range(T):
            t_glob = t + reg_low
            mass_all[idx, t] = np.arccosh((jack(mirtr[t_glob], i) + jack(mirtr[t_glob + 2], i)) / (2 * jack(mirtr[t_glob + 1], i)))
        idx += 1

    # 2. Mittelwert über Jackknife-Blöcke
    mean_mass = np.mean(mass_all, axis=0)

    # 3. Kovarianzmatrix berechnen
    covmat = np.zeros((T, T))
    for t1 in range(T):
        for t2 in range(T):
            covmat[t1, t2] = np.sum((mass_all[:, t1] - mean_mass[t1]) * (mass_all[:, t2] - mean_mass[t2])) / (njack - 1)

    # 4. Inverse zurückgeben
    return np.linalg.inv(covmat)



# Get Pat

parser = argparse.ArgumentParser()
parser.add_argument('--ensemble', type=str, required=True)
parser.add_argument('--particle', type=str, required=True)
parser.add_argument('--nsq', type=int, required=True)
parser.add_argument('--cmass_index', type=int, required=True)
parser.add_argument('--unfrozen', action='store_true', help='Use unfrozen fit (jackknife-dependent covariance)')
args = parser.parse_args()

# Use the parsed arguments
Ensemble = args.ensemble
particle = args.particle
nsq = args.nsq
cmass_index = args.cmass_index

if Ensemble == 'F1S':
    reg_low=19#18
    reg_up=25#25
elif Ensemble in ['M1', 'M2', 'M3']:
    reg_low=19#17#12
    reg_up=25
elif Ensemble in ['C1', 'C2']:
    reg_low=14#10
    reg_up=25

reg_up=reg_up+1

cmass=Ens.getCmass(Ensemble)[cmass_index]

#Get Path
fpath='../Data/{}/BsDsStar_{}_2pt{}.h5'.format(Ensemble,Ensemble,particle)
f=h5py.File(fpath, "r")
path='../Data/{}/2pt/'.format(Ensemble,Ensemble,particle)
configs,dt,ti,L= Ens.getEns(Ensemble)
m,csw,zeta=Ens.getRHQparams(Ensemble)
sm=Ens.getSM(Ensemble)
smass=Ens.getSmass(Ensemble)

if particle == 'Bs':
    #bs=f['hl_SM10.36_PT_0.025/operator_Gamma5/n2_{}/data'.format(nsq)]
    bs=f["/hl_SM{}_PT_{}_m{}_csw{}_zeta{}/operator_Gamma5/n2_0/data".format(sm,smass,m,csw,zeta)]
    mir = np.zeros(shape=(configs, int(ti/2+1)))
    for k in range(configs):
        mir[k][0]=np.real(np.mean(bs[k][:][0]))
        for j in range(int(ti/2)):
            mir[k][j+1]=(np.mean(np.real(bs), axis=1)[k,j+1]+np.mean(np.real(bs), axis=1)[k,ti-1-j])/2

else:
    dsx=f["/cl_SM{}_PT_{}/c{}/operator_GammaX/n2_{}/data".format(sm,smass,cmass,nsq)]
    dsy=f["/cl_SM{}_PT_{}/c{}/operator_GammaY/n2_{}/data".format(sm,smass,cmass,nsq)]
    dsz=f["/cl_SM{}_PT_{}/c{}/operator_GammaZ/n2_{}/data".format(sm,smass,cmass,nsq)]
    #dsx=f["/cl_SM10.36_PT_0.025/c{}/operator_GammaX/n2_{}/data".format(cmass,nsq)]
    #dsy=f["/cl_SM10.36_PT_0.025/c{}/operator_GammaY/n2_{}/data".format(cmass,nsq)]
    #dsz=f["/cl_SM10.36_PT_0.025/c{}/operator_GammaZ/n2_{}/data".format(cmass,nsq)]

    #folding:
    mir = np.zeros(shape=(configs, int(ti/2+1)))
    for k in range(configs):
        mir[k][0]=(np.real(np.mean(dsx[k][:][0]))+np.real(np.mean(dsy[k][:][0]))+np.real(np.mean(dsz[k][:][0])))/3
        for j in range(int(ti/2)):
            mir[k][j+1]=((np.mean(np.real(dsx), axis=1)[k,j+1]+np.mean(np.real(dsx), axis=1)[k,ti-1-j])/2+(np.mean(np.real(dsy), axis=1)[k,j+1]+np.mean(np.real(dsy), axis=1)[k,ti-1-j])/2+(np.mean(np.real(dsz), axis=1)[k,j+1]+np.mean(np.real(dsz), axis=1)[k,ti-1-j])/2)/3


# Correlator
res= np.zeros(int(ti/2+1))
error=np.zeros(int(ti/2+1))
for j in range(int(ti/2+1)):
    res[j]=exp_val(extract(mir,j))
    r=0
    for i in range(configs):
        r=r+(jack(extract(mir,j),i)-res[j])**2
    error[j]=np.sqrt((configs-1)/configs*r)

#Effective Mass
mirtr=np.transpose(mir)  
mass=np.zeros(int(ti/2-1))
errors=np.zeros(int(ti/2-1))
for i in range(int(ti/2-1)):
    mass[i]=(cmath.acosh((res[i]+res[i+2])/(2*res[i+1])+0j)).real
    x=0
    for j in range(configs):    
        x=x+(np.arccosh((jack(mirtr[i],j)+jack(mirtr[i+2],j))/(2*jack(mirtr[i+1],j)))-mass[i])**2
    errors[i]=(np.sqrt((configs-1)/configs*x))
    
    
#Save results to csv
df1 = pd.DataFrame(columns=['Correlator','Error'])
df1['Correlator']=res
df1['Error']=error

df2 = pd.DataFrame(columns=['EffectiveMass','Error'])
df2['EffectiveMass']=mass
df2['Error']=errors    

if particle == 'Bs':
    df1.to_csv(path+'Corr-Bs.csv', sep='\t')
    df2.to_csv(path+'Mass-Bs.csv', sep='\t')
else:
    df1.to_csv(path+'Corr-Ds{}-{}.csv'.format(cmass,nsq), sep='\t')
    df2.to_csv(path+'Mass-Ds{}-{}.csv'.format(cmass,nsq), sep='\t')


###############################################################################

#Covarianze matrix 

covmat=np.zeros(shape=(int(reg_up-reg_low),int(reg_up-reg_low)))
for t1 in range(int(reg_up-reg_low)):
    for t2 in range(int(reg_up-reg_low)):
        x=0
        for i in range(configs):  
            x=x+(np.arccosh((jack(mirtr[t1+reg_low],i)+jack(mirtr[t1+2+reg_low],i))/(2*jack(mirtr[t1+1+reg_low],i)))-mass[t1+reg_low])*(np.arccosh((jack(mirtr[t2+reg_low],i)+jack(mirtr[t2+2+reg_low],i))/(2*jack(mirtr[t2+1+reg_low],i)))-mass[t2+reg_low])
        covmat[t1][t2]=(configs-1)/configs*x
        covmat[t2][t1]=(configs-1)/configs*x  


invcovmat=np.linalg.inv(covmat)        
def chi(a):
    return np.dot(np.transpose([i-a for i in mass[reg_low:reg_up]]),np.matmul(invcovmat,[i-a for i in mass[reg_low:reg_up]]))

mbar=minimize(chi,0.1,method='Nelder-Mead', tol=1e-8)

def jackmass(t1,i):
    return np.arccosh((jack(mirtr[t1],i)+jack(mirtr[t1+2],i))/(2*jack(mirtr[t1+1],i)))

def chijack(a,k):
    return np.dot(np.transpose([jackmass(i+reg_low,k)-a for i in range(int(reg_up-reg_low))]),np.matmul(invcovmat,[jackmass(i+reg_low,k)-a for i in range(int(reg_up-reg_low))]))


#Std Deviation for all jakcknife blocks
jblocks=np.zeros(configs)
h=0
for i in range(configs):
    tmp=minimize(chijack,0.1,args=(i),method='Nelder-Mead', tol=1e-8).x[0]
    jblocks[i]=tmp
    h=h+(tmp-mbar.x[0])**2
sigma=np.sqrt((configs-1)/configs*h)



jblocks = np.zeros(configs)

if args.unfrozen:
    for i in range(configs):
        # Neue effektive Massen ohne Block i
        mass_block = np.zeros(int(ti/2 - 1))
        for t in range(int(ti/2 - 1)):
            mass_block[t] = np.arccosh((jack(mirtr[t], i) + jack(mirtr[t+2], i)) / (2 * jack(mirtr[t+1], i)))

       # print(mass_block)
        # Lokale Kovarianzmatrix
        try:
            #inv_cov_block = build_covmat_from_jackblock(mirtr, mass_block, reg_low, reg_up, i)
            inv_cov_block = build_covmat_from_jackblock(mirtr, reg_low, reg_up, exclude_idx=i)
        except np.linalg.LinAlgError:
            print(f"Singular matrix at jackknife block {i}, skipping...")
            continue

        # Chi^2 Funktion
        def chijack_unfrozen(a):
            delta = mass_block[reg_low:reg_up] - a
            return np.dot(delta, np.matmul(inv_cov_block, delta))

        # Minimieren
        tmp = minimize(chijack_unfrozen, 0.1, method='Nelder-Mead', tol=1e-8).x[0]
        jblocks[i] = tmp
    # NEU: mbar_unfrozen = Jackknife-Mittelwert
    mbar_unfrozen = np.mean(jblocks)
    sigma = np.sqrt((configs - 1)/configs * np.sum((jblocks - mbar_unfrozen)**2))

else:
    h = 0
    for i in range(configs):
        tmp = minimize(chijack, 0.1, args=(i), method='Nelder-Mead', tol=1e-8).x[0]
        jblocks[i] = tmp
        h += (tmp - mbar.x[0])**2
    sigma = np.sqrt((configs - 1)/configs * h)
    mbar_unfrozen = mbar.x[0]  # für einheitliche Schreibweise unten



df4 = pd.DataFrame(columns=['EffectiveMass'])
df4['EffectiveMass']=jblocks   

plt.figure(constrained_layout=True)
plt.xlabel('Time Steps')
plt.ylabel('Eff. Energy')
plt.errorbar(list(range(47))[1:30], mass[1:30], yerr=errors[1:30],fmt='x')
plt.axhline(y = mbar.x[0], color = 'r', linestyle = 'dashed', label = "red line")
plt.fill_between(list(range(47))[reg_low:reg_up], mbar.x[0]+sigma, mbar.x[0]-sigma, color='r',alpha=0.2)
plt.yscale('log')

df3 = pd.DataFrame(columns=['EffectiveMass','Error','RegUp','RegLow'])
df3['EffectiveMass']=mbar.x
df3['Error']=sigma  
df3['RegUp']=reg_up
df3['RegLow']=reg_low    

df5 = pd.DataFrame(columns=['pval'])
chisq=mbar.fun
pval= pvalue(mbar.fun,reg_up-reg_low)
df5 = pd.DataFrame({'pval': [pval], 'chisq': [chisq]})

if particle == 'Bs':
    df4.to_csv(path+'Bs-blocks-uf.csv', sep='\t')
    plt.savefig(path+'Zoom-Bs-Reg-uf.pdf')
    df3.to_csv(path+'BsResult-uf.csv', sep='\t')
    df5.to_csv(path+'pval-Bs-uf.csv', sep='\t')
else:
    df4.to_csv(path+'Ds{}-nsq{}-blocks-uf.csv'.format(cmass,nsq), sep='\t')
    plt.savefig(path+'Zoom-Ds{}-Reg-{}-uf.pdf'.format(cmass,nsq))
    df3.to_csv(path+'Ds{}Result-{}-uf.csv'.format(cmass,nsq), sep='\t')
    df5.to_csv(path+'pval-Ds{}-{}-uf.csv'.format(cmass,nsq), sep='\t')

