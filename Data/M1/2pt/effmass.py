### Current error. mass has NaN-> cosh for vaslues lower than one

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cmath
import pandas as pd
from scipy.optimize import minimize
import scipy
import sys 
import os

def pvalue(chi2, dof):
    r"""Compute the $p$-value corresponding to a $\chi^2$ with `dof` degrees
    of freedom."""
    return 1 - scipy.stats.chi2.cdf(chi2, dof)
 

# Add the Code directory to the system path
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Code'))
sys.path.append(code_dir)

import Ensemble as Ens

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


#f = h5py.File("../BsDsStar_M1_2ptBs.h5", "r")
f = h5py.File("../BsDsStar_M1_2ptDs.h5", "r")

#########
nsq=3
ensemble='M1'


cmass=Ens.getCmass(ensemble)[2]
configs,dt,ti,L= Ens.getEns(ensemble)


bsn0=f["/cl_SM10.36_PT_0.025/c{}/operator_GammaX/n2_{}/data".format(cmass,nsq)]
bsn0y=f["/cl_SM10.36_PT_0.025/c{}/operator_GammaY/n2_{}/data".format(cmass,nsq)]
bsn0z=f["/cl_SM10.36_PT_0.025/c{}/operator_GammaZ/n2_{}/data".format(cmass,nsq)]
'''


bsn0=f['hl_SM10.36_PT_0.025/operator_Gamma5/n2_0/data']
bsn0y=f['hl_SM10.36_PT_0.025/operator_Gamma5/n2_0/data']
bsn0z=f['hl_SM10.36_PT_0.025/operator_Gamma5/n2_0/data']
'''
'''
bsn0=f["/hl_SM10.36_PT_0.025_m3.49_csw3.07_zeta1.76/operator_Gamma5/n2_0/data"]
bsn0y=f["/hl_SM10.36_PT_0.025_m3.49_csw3.07_zeta1.76/operator_Gamma5/n2_0/data"]
bsn0z=f["/hl_SM10.36_PT_0.025_m3.49_csw3.07_zeta1.76/operator_Gamma5/n2_0/data"]
'''

#ti=64
#configs=1636

print(bsn0[0][:])

#folding

mir = np.zeros(shape=(configs, int(ti/2+1)))
miry = np.zeros(shape=(configs, int(ti/2+1)))
mirz = np.zeros(shape=(configs, int(ti/2+1)))
for k in range(configs):
    mir[k][0]=(np.real(bsn0[k][0][0]))
    miry[k][0]=(np.real(bsn0y[k][0][0]))
    mirz[k][0]=(np.real(bsn0z[k][0][0]))
    for j in range(int(ti/2)):
        mir[k][j+1]=((np.real(bsn0[k][0][j+1])+np.real(bsn0[k][0][ti-1-j]))/2+(np.real(bsn0y[k][0][j+1])+np.real(bsn0y[k][0][ti-1-j]))/2+(np.real(bsn0z[k][0][j+1])+np.real(bsn0z[k][0][ti-1-j]))/2)/3
        #miry[k][j+1]=(np.real(bsn0y[k][0][j+1])+np.real(bsn0y[k][0][ti-1-j]))/2
        #mirz[k][j+1]=(np.real(bsn0z[k][0][j+1])+np.real(bsn0z[k][0][ti-1-j]))/2

'''
mir = np.zeros(shape=(configs, int(ti/2+1)))
miry = np.zeros(shape=(configs, int(ti/2+1)))
mirz = np.zeros(shape=(configs, int(ti/2+1)))
for k in range(configs):
    mir[k][0]=(np.real(bsn0[k][0]))
    miry[k][0]=(np.real(bsn0y[k][0]))
    mirz[k][0]=(np.real(bsn0z[k][0]))
    for j in range(int(ti/2)):
        mir[k][j+1]=(np.real(bsn0[k][j+1])+np.real(bsn0[k][64-1-j]))/2
        miry[k][j+1]=(np.real(bsn0y[k][j+1])+np.real(bsn0y[k][64-1-j]))/2
        mirz[k][j+1]=(np.real(bsn0z[k][j+1])+np.real(bsn0z[k][64-1-j]))/2
'''


res= np.zeros(int(ti/2+1))
error=np.zeros(int(ti/2+1))
 
for j in range(int(ti/2+1)):
    #res[j]=(exp_val(extract(mir,j))+exp_val(extract(miry,j))+exp_val(extract(mirz,j)))/3
    res[j]=exp_val(extract(mir,j))
    r=0
    for i in range(configs):
        #r=r+((jack(extract(mir,j),i)+jack(extract(miry,j),i)+jack(extract(mirz,j),i))/3-res[j])**2
        r=r+(jack(extract(mir,j),i)-res[j])**2
    error[j]=np.sqrt((configs-1)/configs*r)


mirtr=np.transpose(mir)  
#mirtry=np.transpose(miry)  
#mirtrz=np.transpose(mirz)  

#f = plt.figure(figsize=(10,4))
#plt.subplots_adjust(bottom=0.5)
#ax = f.add_subplot(121)
#ax2 = f.add_subplot(122)

#ax.set_xlabel('Time Steps')
#ax.set_ylabel('2pt fct')
#ax.errorbar(list(range(int(ti/2+1))), res, yerr=error,fmt='1')
#ax.set_yscale('log')

mass=np.zeros(int(ti/2-1))
for i in range(int(ti/2-1)):
    mass[i]=(cmath.acosh((res[i]+res[i+2])/(2*res[i+1])+0j)).real
    
    
errors=np.zeros(int(ti/2-1))
for j in range(int(ti/2-1)):
    x=0
    for i in range(configs):    
        #x=x+(np.arccosh(((jack(mirtr[j],i)+jack(mirtry[j],i)+jack(mirtrz[j],i))/3+(jack(mirtr[j+2],i)+jack(mirtry[j+2],i)+jack(mirtrz[j+2],i))/3)/(2*(jack(mirtr[j+1],i)+jack(mirtry[j+1],i)+jack(mirtrz[j+1],i))/3))-mass[j])**2
        x=x+(np.arccosh((jack(mirtr[j],i)+jack(mirtr[j+2],i))/(2*jack(mirtr[j+1],i)))-mass[j])**2
    errors[j]=(np.sqrt((configs-1)/configs*x))

    
#Save reuslts to csv
df1 = pd.DataFrame(columns=['Correlator','Error'])
df1['Correlator']=res
df1['Error']=error
df1.to_csv('Corr-Ds{}-{}.csv'.format(cmass,nsq), sep='\t')
#df1.to_csv('Corr-Bs.csv', sep='\t')

df2 = pd.DataFrame(columns=['EffectiveMass','Error'])
df2['EffectiveMass']=mass
df2['Error']=errors    
#df2.to_csv('Mass-Bs.csv', sep='\t')
df2.to_csv('Mass-Ds{}-{}.csv'.format(cmass,nsq), sep='\t')

print('Test1')

###############################################################################

reg_low=12
reg_up=25

#Covarianze matrix (without prefactor, not squarrooted)
cut=ti/2-1-reg_up
cut1=ti/2+1-reg_up
covmat=np.zeros(shape=(int(ti/2-1-reg_low-cut),int(ti/2-1-reg_low-cut)))
for t1 in range(int(ti/2-1-reg_low-cut)):
    for t2 in range(int(ti/2-1-reg_low-cut)):
        x=0
        for i in range(configs):  
            #x=x+(np.arccosh(((jack(mirtr[t1+reg_low],i)+jack(mirtry[t1+reg_low],i)+jack(mirtrz[t1+reg_low],i))/3+(jack(mirtr[t1+2+reg_low],i)+jack(mirtry[t1+2+reg_low],i)+jack(mirtrz[t1+2+reg_low],i))/3)/(2*(jack(mirtr[t1+1+reg_low],i)+jack(mirtry[t1+1+reg_low],i)+jack(mirtrz[t1+1+reg_low],i))/3))-mass[t1+reg_low])*(np.arccosh(((jack(mirtr[t2+reg_low],i)+jack(mirtry[t2+reg_low],i)+jack(mirtrz[t2+reg_low],i))/3+(jack(mirtr[t2+2+reg_low],i)+jack(mirtry[t2+2+reg_low],i)+jack(mirtrz[t2+2+reg_low],i))/3)/(2*(jack(mirtr[t2+1+reg_low],i)+jack(mirtry[t2+1+reg_low],i)+jack(mirtrz[t2+1+reg_low],i))/3))-mass[t2+reg_low])
            x=x+(np.arccosh((jack(mirtr[t1+reg_low],i)+jack(mirtr[t1+2+reg_low],i))/(2*jack(mirtr[t1+1+reg_low],i)))-mass[t1+reg_low])*(np.arccosh((jack(mirtr[t2+reg_low],i)+jack(mirtr[t2+2+reg_low],i))/(2*jack(mirtr[t2+1+reg_low],i)))-mass[t2+reg_low])
        covmat[t1][t2]=x
        covmat[t2][t1]=x  
        
print('test')
invcovmat=np.linalg.inv(covmat)        
def chi(a):
    return (configs)/(configs-1)*np.dot(np.transpose([i-a for i in mass[reg_low:reg_up]]),np.matmul(invcovmat,[i-a for i in mass[reg_low:reg_up]]))

mbar=minimize(chi,0.1,method='Nelder-Mead', tol=1e-8)


def jackmass(t1,i):
    #return np.arccosh(((jack(mirtr[t1],i)+jack(mirtry[t1],i)+jack(mirtrz[t1],i))/3+(jack(mirtr[t1+2],i)+jack(mirtry[t1+2],i)+jack(mirtrz[t1+2],i))/3)/(2*(jack(mirtr[t1+1],i)+jack(mirtry[t1+1],i)+jack(mirtrz[t1+1],i))/3))
    return np.arccosh((jack(mirtr[t1],i)+jack(mirtr[t1+2],i))/(2*jack(mirtr[t1+1],i)))

def chijack(a,k):
    return np.dot(np.transpose([jackmass(i+reg_low,k)-a for i in range(int(ti/2-1-reg_low-cut))]),np.matmul(invcovmat,[jackmass(i+reg_low,k)-a for i in range(int(ti/2-1-reg_low-cut))]))


#Std Deviation for all jakcknife blocks
jblocks=np.zeros(configs)
h=0
for i in range(configs):
    tmp=minimize(chijack,0.1,args=(i),method='Nelder-Mead', tol=1e-6).x[0]
    jblocks[i]=tmp
    h=h+(tmp-mbar.x[0])**2
sigma=np.sqrt((configs-1)/(configs)*h)

df4 = pd.DataFrame(columns=['EffectiveMass'])
df4['EffectiveMass']=jblocks   
df4.to_csv('Ds{}-nsq{}-blocks.csv'.format(cmass,nsq), sep='\t')
#df4.to_csv('Bs-blocks.csv', sep='\t')

print(mbar,sigma)

plt.figure(constrained_layout=True)
plt.xlabel('Time Steps')
plt.ylabel('Eff. Energy')
plt.errorbar(list(range(47))[1:30], mass[1:30], yerr=errors[1:30],fmt='x')
plt.axhline(y = mbar.x[0], color = 'r', linestyle = 'dashed', label = "red line")
plt.fill_between(list(range(47))[reg_low:reg_up], mbar.x[0]+sigma, mbar.x[0]-sigma, color='r',alpha=0.2)
plt.yscale('log')
plt.savefig('Zoom-Bs-Reg.pdf')
#plt.savefig('Zoom-Ds{}-Reg-{}.pdf'.format(cmass,nsq))

df3 = pd.DataFrame(columns=['EffectiveMass','Error','RegUp','RegLow'])
df3['EffectiveMass']=mbar.x
df3['Error']=sigma  
df3['RegUp']=reg_up
df3['RegLow']=reg_low    
#df3.to_csv('BsResult.csv', sep='\t')
df3.to_csv('Ds{}Result-{}.csv'.format(cmass,nsq), sep='\t')



df4 = pd.DataFrame(columns=['pval'])
df4['pval']=pvalue(mbar.fun,reg_up-reg_low)
df4.to_csv('pval-Ds{}-{}.csv'.format(cmass,nsq), sep='\t')
#df4.to_csv('pval-Bs{}-{}.csv'.format(cmass,nsq), sep='\t')










        
