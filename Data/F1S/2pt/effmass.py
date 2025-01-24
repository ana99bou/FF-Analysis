### Current error. mass has NaN-> cosh for vaslues lower than one

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cmath
import pandas as pd
from scipy.optimize import minimize

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


f = h5py.File("BsDsStar-nsq5n.h5", "r")
'''
bsn0=f["/CHARM_SM12.14_PT_s0.02144/c0.248/operator_GammaX/2_1_0/data"]
bsn0y=f["/CHARM_SM12.14_PT_s0.02144/c0.248/operator_GammaY/2_1_0/data"]
bsn0z=f["/CHARM_SM12.14_PT_s0.02144/c0.248/operator_GammaZ/2_1_0/data"]
'''
bsn0=f['/rhq_m2.42_csw2.68_zeta1.52_SM12.14_PT_s0.02144/operator_Gamma5/0_0_0/data']
bsn0y=f['/rhq_m2.42_csw2.68_zeta1.52_SM12.14_PT_s0.02144/operator_Gamma5/0_0_0/data']
bsn0z=f['/rhq_m2.42_csw2.68_zeta1.52_SM12.14_PT_s0.02144/operator_Gamma5/0_0_0/data']

ti=96
configs=98


#folding
mir = np.zeros(shape=(98, 49))
miry = np.zeros(shape=(98, 49))
mirz = np.zeros(shape=(98, 49))
for k in range(98):
    tem=0
    temy=0
    temz=0
    for i in range(24): 
        tem=tem+np.real(bsn0[k][i][0])
        temy=temy+np.real(bsn0y[k][i][0])
        temz=temz+np.real(bsn0z[k][i][0])
    mir[k][0]=tem/24
    miry[k][0]=temy/24
    mirz[k][0]=temz/24
    for j in range(48):
        tem=0
        temy=0
        temz=0
        for i in range(24): 
            tem=tem+(np.real(bsn0[k][i][j+1])+np.real(bsn0[k][i][96-1-j]))/2
            temy=temy+(np.real(bsn0y[k][i][j+1])+np.real(bsn0y[k][i][96-1-j]))/2
            temz=temz+(np.real(bsn0z[k][i][j+1])+np.real(bsn0z[k][i][96-1-j]))/2
        mir[k][j+1]=tem/24
        miry[k][j+1]=temy/24
        mirz[k][j+1]=temz/24

res= np.zeros(49)
error=np.zeros(49)
 
for j in range(49):
    res[j]=(exp_val(extract(mir,j))+exp_val(extract(miry,j))+exp_val(extract(mirz,j)))/3
    r=0
    for i in range(configs):
        r=r+((jack(extract(mir,j),i)+jack(extract(miry,j),i)+jack(extract(mirz,j),i))/3-res[j])**2
    error[j]=np.sqrt((configs-1)/configs*r)


mirtr=np.transpose(mir)  
mirtry=np.transpose(miry)  
mirtrz=np.transpose(mirz)  

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
        x=x+(np.arccosh(((jack(mirtr[j],i)+jack(mirtry[j],i)+jack(mirtrz[j],i))/3+(jack(mirtr[j+2],i)+jack(mirtry[j+2],i)+jack(mirtrz[j+2],i))/3)/(2*(jack(mirtr[j+1],i)+jack(mirtry[j+1],i)+jack(mirtrz[j+1],i))/3))-mass[j])**2
    errors[j]=(np.sqrt((configs-1)/configs*x))

    
#Save reuslts to csv
df1 = pd.DataFrame(columns=['Correlator','Error'])
df1['Correlator']=res
df1['Error']=error
df1.to_csv('Corr-Bs.csv', sep='\t')

df2 = pd.DataFrame(columns=['EffectiveMass','Error'])
df2['EffectiveMass']=mass
df2['Error']=errors    
df2.to_csv('Mass-Bs.csv', sep='\t')



###############################################################################

reg_low=18
reg_up=25

#Covarianze matrix (without prefactor, not squarrooted)
cut=ti/2-1-reg_up
cut1=ti/2+1-reg_up
covmat=np.zeros(shape=(int(ti/2-1-reg_low-cut),int(ti/2-1-reg_low-cut)))
for t1 in range(int(ti/2-1-reg_low-cut)):
    for t2 in range(int(ti/2-1-reg_low-cut)):
        x=0
        for i in range(configs):  
            x=x+(np.arccosh(((jack(mirtr[t1+reg_low],i)+jack(mirtry[t1+reg_low],i)+jack(mirtrz[t1+reg_low],i))/3+(jack(mirtr[t1+2+reg_low],i)+jack(mirtry[t1+2+reg_low],i)+jack(mirtrz[t1+2+reg_low],i))/3)/(2*(jack(mirtr[t1+1+reg_low],i)+jack(mirtry[t1+1+reg_low],i)+jack(mirtrz[t1+1+reg_low],i))/3))-mass[t1+reg_low])*(np.arccosh(((jack(mirtr[t2+reg_low],i)+jack(mirtry[t2+reg_low],i)+jack(mirtrz[t2+reg_low],i))/3+(jack(mirtr[t2+2+reg_low],i)+jack(mirtry[t2+2+reg_low],i)+jack(mirtrz[t2+2+reg_low],i))/3)/(2*(jack(mirtr[t2+1+reg_low],i)+jack(mirtry[t2+1+reg_low],i)+jack(mirtrz[t2+1+reg_low],i))/3))-mass[t2+reg_low])
        covmat[t1][t2]=x
        covmat[t2][t1]=x  
        
        
def chi(a):
    return (configs-1-reg_low-cut)/(configs-reg_low-cut)*np.dot(np.transpose([i-a for i in mass[reg_low:reg_up]]),np.matmul(np.linalg.inv(covmat),[i-a for i in mass[reg_low:reg_up]]))

mbar=minimize(chi,0.1,method='Nelder-Mead', tol=1e-6)


def jackmass(t1,i):
    return np.arccosh(((jack(mirtr[t1],i)+jack(mirtry[t1],i)+jack(mirtrz[t1],i))/3+(jack(mirtr[t1+2],i)+jack(mirtry[t1+2],i)+jack(mirtrz[t1+2],i))/3)/(2*(jack(mirtr[t1+1],i)+jack(mirtry[t1+1],i)+jack(mirtrz[t1+1],i))/3))

def chijack(a,k):
    return np.dot(np.transpose([jackmass(i+reg_low,k)-a for i in range(int(ti/2-1-reg_low-cut))]),np.matmul(np.linalg.inv(covmat),[jackmass(i+reg_low,k)-a for i in range(int(ti/2-1-reg_low-cut))]))


jblocks=np.zeros(configs)

#Std Deviation for all jakcknife blocks
h=0
for i in range(configs):
    tmp=minimize(chijack,0.1,args=(i),method='Nelder-Mead', tol=1e-6).x[0]
    jblocks[i]=tmp
    h=h+(minimize(chijack,0.1,args=(i),method='Nelder-Mead', tol=1e-6).x[0]-mbar.x[0])**2
sigma=np.sqrt((configs-1-reg_low-cut)/(configs-reg_low-cut)*h)

df4 = pd.DataFrame(columns=['EffectiveMass'])
df4['EffectiveMass']=jblocks   
df4.to_csv('Bs-blocks.csv', sep='\t')


print(mbar,sigma)

plt.figure(constrained_layout=True)
plt.xlabel('Time Steps')
plt.ylabel('Eff. Energy')
plt.errorbar(list(range(47))[10:46], mass[10:46], yerr=errors[10:46],fmt='x')
plt.axhline(y = mbar.x[0], color = 'r', linestyle = 'dashed', label = "red line")
plt.fill_between(list(range(47))[reg_low:reg_up], mbar.x[0]+sigma, mbar.x[0]-sigma, color='r',alpha=0.2)
plt.yscale('log')
plt.savefig('Zoom-Bs-Reg.pdf')

df3 = pd.DataFrame(columns=['EffectiveMass','Error','RegUp','RegLow'])
df3['EffectiveMass']=mbar.x
df3['Error']=sigma  
df3['RegUp']=reg_up
df3['RegLow']=reg_low    
df3.to_csv('BsResult.csv', sep='\t')










        