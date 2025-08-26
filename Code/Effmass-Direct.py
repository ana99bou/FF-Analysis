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
from iminuit import Minuit


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


Ensemble = 'F1S'  # Example value, replace with actual input
particle = 'Ds'  # Example value, replace with actual input
nsq = 0  # Example value, replace with actual input
cmass_index = 1  # Example value, replace with actual input

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
if Ensemble == 'F1S':
    reg_low=19#18
    reg_up=25
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
mirtr=np.transpose(mir)  

    
#Save results to csv
df1 = pd.DataFrame(columns=['Correlator','Error'])
df1['Correlator']=res
df1['Error']=error



###############################################################################

#Covarianze matrix 
masstmp=np.log(res[reg_low]/res[reg_low+1])
#masstmp=0.7341573429107688
print(masstmp)
a0tmp=res[reg_low]/(np.exp(-masstmp*reg_low)+np.exp(-masstmp*(ti/2+1-reg_low)))
#a0tmp=res[20]/(np.exp(-masstmp*20)+np.exp(-masstmp*(ti/2+1-20)))#-0.0000562341e-14

covmat=np.zeros(shape=(int(reg_up-reg_low),int(reg_up-reg_low)))
for t1 in range(int(reg_up-reg_low)):
    for t2 in range(int(reg_up-reg_low)):
        x=0
        for i in range(configs):  
            x=x+(jack(mirtr[t1+reg_low],i)-res[t1+reg_low])*(jack(mirtr[t2+reg_low],i)-res[t2+reg_low])
        covmat[t1][t2]=(configs-1)/configs*x
        covmat[t2][t1]=(configs-1)/configs*x  

invcovmat=np.linalg.inv(covmat) 
lst=list(range(reg_low, reg_up))

'''
def chi(params):
    a1, a2 = params
    return np.dot(np.transpose([res[i]-(a1*(np.exp(-a2*i)+np.exp(-a2*(ti-i)))) for i in lst]),np.matmul(invcovmat,[res[i]-(a1*(np.exp(-a2*i)+np.exp(-a2*(ti-i)))) for i in lst]))

mbar=minimize(chi,[a0tmp, masstmp], method='Nelder-Mead', tol=1e-8)    
'''

def chi(a1, a2):
    return np.dot(
        np.transpose([res[i] - (a1 * (np.exp(-a2 * i) + np.exp(-a2 * (ti - i)))) for i in lst]),
        np.matmul(invcovmat, [res[i] - (a1 * (np.exp(-a2 * i) + np.exp(-a2 * (ti - i)))) for i in lst])
    )
mbar = Minuit(chi, a1=1, a2=1)
mbar.errordef = 1  # For least-squares (χ²), errordef = 1
mbar.migrad()      # Run the minimization


'''
print('Initial guess: a1 = {}, a2 = {}'.format(a0tmp, masstmp))
print('Optimized parameters: a1 = {}, a2 = {}'.format(mbar.x[0], mbar.x[1]))
print('Chi-squared:', mbar.fun)
'''

print('Initial guess: a1 = {}, a2 = {}'.format(a0tmp, masstmp))
print("Optimized a1:", mbar.values["a1"])
print("Optimized a2 (mass):", mbar.values["a2"])
print("Minimum chi²:", mbar.fval)

best_a1 = mbar.values["a1"]
best_a2 = mbar.values["a2"]

def jackmass(t1,i):
    return jack(mirtr[t1],i)

'''
def chijack(params,k):
    a1, a2 = params
    return np.dot(np.transpose([jackmass(i,k)-(a1*(np.exp(-a2*(i))+np.exp(-a2*(ti-(i))))) for i in lst]),np.matmul(invcovmat,[jackmass(i,k)-(a1*(np.exp(-a2*(i))+np.exp(-a2*(ti-(i))))) for i in lst]))
'''

def chijack(a1, a2, k):
    return np.dot(
        np.transpose([jackmass(i, k) - (a1 * (np.exp(-a2 * i) + np.exp(-a2 * (ti - i)))) for i in lst]),
        np.matmul(invcovmat, [jackmass(i, k) - (a1 * (np.exp(-a2 * i) + np.exp(-a2 * (ti - i)))) for i in lst])
    )


#Std Deviation for all jakcknife blocks
jblocks=np.zeros(configs)
h=0

'''
for i in range(configs):
    tmp=minimize(chijack,[a0tmp, masstmp],args=(i), method='Nelder-Mead').x[1]
    jblocks[i]=tmp
    h=h+(tmp-mbar.x[1])**2
sigma=np.sqrt((configs-1)/configs*h)
'''

for i in range(configs):
    mj = Minuit(lambda a1, a2: chijack(a1, a2, i), a1=a0tmp, a2=masstmp)
    mj.errordef = 1
    mj.migrad()
    jblocks[i] = mj.values["a2"]
    h += (mj.values["a2"] - best_a2) ** 2
sigma = np.sqrt((configs - 1) / configs * h)

print('Fehler:',sigma)


plt.figure(figsize=(8,6))

# Full time range
full_range = list(range(ti // 2 + 1))

# Plot all correlator points with errors
plt.errorbar(full_range, [res[t] for t in full_range], yerr=[error[t] for t in full_range], fmt='o', label='Correlator', capsize=3)

# Full fit function over the whole range
full_fit = [best_a1 * (np.exp(-best_a2 * t) + np.exp(-best_a2 * (ti - t))) for t in full_range]
plt.plot(full_range, full_fit, color='red', label='Fit: $A_1 e^{-A_2 t} + A_1 e^{-A_2 (T - t)}$')

plt.yscale('log')  
plt.xlabel('Time')
plt.ylabel('2pt Correlator')
#plt.title(f'{particle} Correlator Fit')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('Fit.pdf')

df3 = pd.DataFrame([{
    'Amplitude': best_a1,
    'Mass': best_a2,
    'Error': sigma,
    'RegUp': reg_up,
    'RegLow': reg_low
}])


df5 = pd.DataFrame(columns=['pval'])
chisq=mbar.fval
pval= pvalue(mbar.fval,reg_up-reg_low)
df5 = pd.DataFrame({'pval': [pval], 'chisq': [chisq]})

if particle == 'Bs':
    #df4.to_csv(path+'Bs-blocks.csv', sep='\t')
    plt.savefig(path+'Direct-Zoom-Bs-Reg.pdf')
    df3.to_csv(path+'Direct-BsResult.csv', sep='\t')
    df5.to_csv(path+'Direct-pval-Bs.csv', sep='\t')
else:
    #df4.to_csv(path+'Ds{}-nsq{}-blocks.csv'.format(cmass,nsq), sep='\t')
    plt.savefig(path+'Direct-Zoom-Ds{}-Reg-{}.pdf'.format(cmass,nsq))
    df3.to_csv(path+'Direct-Ds{}Result-{}.csv'.format(cmass,nsq), sep='\t')
    df5.to_csv(path+'Direct-pval-Ds{}-{}.csv'.format(cmass,nsq), sep='\t')




'''
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
    df4.to_csv(path+'Bs-blocks.csv', sep='\t')
    plt.savefig(path+'Zoom-Bs-Reg.pdf')
    df3.to_csv(path+'BsResult.csv', sep='\t')
    df5.to_csv(path+'pval-Bs.csv', sep='\t')
else:
    df4.to_csv(path+'Ds{}-nsq{}-blocks.csv'.format(cmass,nsq), sep='\t')
    plt.savefig(path+'Zoom-Ds{}-Reg-{}.pdf'.format(cmass,nsq))
    df3.to_csv(path+'Ds{}Result-{}.csv'.format(cmass,nsq), sep='\t')
    df5.to_csv(path+'pval-Ds{}-{}.csv'.format(cmass,nsq), sep='\t')
'''