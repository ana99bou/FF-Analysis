import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import GetCombs
import Folding
import Ensemble as Ens
import Jackblocks
import Ratio
import Basic
import Regression
 
#########Choose Params
FF='A2'
nsq=5
ensemble='F1S'
cmass=Ens.getCmass(ensemble)[0] #Ens.getCmass(ensemble) gives us an array of the different charm masses for each ens; chose which one
reg_low=18
reg_up=25
##########


# Get strange mass and smearing radius
sm=Ens.getSM(ensemble)
smass=Ens.getSmass(ensemble)


# Ensemble details
nconf,dt,ts,L= Ens.getEns(ensemble)
# Not needed for old F1S
#m,csw,zeta=Ens.getRHQparams(ensemble)

# Read h5 file -> 
##################TODO three h5files for each 3pt and 2pt and get the naming consistent
'''
f3pt = h5py.File("../Data/{}/BsDsStar_{}_3pt.h5".format(ensemble,ensemble), "r")
f2ptDs = h5py.File("../Data/{}/BsDsStar_{}_2ptDs.h5".format(ensemble,ensemble), "r")
f2ptBs = h5py.File("../Data/{}/BsDsStar_{}_2ptBs.h5".format(ensemble,ensemble), "r")
'''

f3pt = h5py.File("../Data/{}/BsDsStar.h5".format(ensemble,ensemble), "r")
f2ptDs = h5py.File("../Data/{}/BsDsStar.h5".format(ensemble,ensemble), "r")
f2ptBs = h5py.File("../Data/{}/BsDsStar.h5".format(ensemble,ensemble), "r")


# Eff. Masses central values
edlist=[pd.read_csv('../Data/{}/2pt/Ds{}Result-0.csv'.format(ensemble,cmass), sep='\t',index_col=0).loc[0,'EffectiveMass'],
        pd.read_csv('../Data/{}/2pt/Ds{}Result-1.csv'.format(ensemble,cmass), sep='\t',index_col=0).loc[0,'EffectiveMass'],
        pd.read_csv('../Data/{}/2pt/Ds{}Result-2.csv'.format(ensemble,cmass), sep='\t',index_col=0).loc[0,'EffectiveMass'],
        pd.read_csv('../Data/{}/2pt/Ds{}Result-3.csv'.format(ensemble,cmass), sep='\t',index_col=0).loc[0,'EffectiveMass'],
        pd.read_csv('../Data/{}/2pt/Ds{}Result-4.csv'.format(ensemble,cmass), sep='\t',index_col=0).loc[0,'EffectiveMass'],
        pd.read_csv('../Data/{}/2pt/Ds{}Result-5.csv'.format(ensemble,cmass), sep='\t',index_col=0).loc[0,'EffectiveMass']]
ed=edlist[nsq]
mb=pd.read_csv('../Data/{}/2pt/BsResult.csv'.format(ensemble), sep='\t',index_col=0).loc[0,'EffectiveMass']
md = edlist[0]

#Get directions of momenta etc
if FF == 'V':
    mom,pref=GetCombs.get_moms_and_prefacs_V()
elif FF == 'A0':
    mom,pref=GetCombs.get_moms_and_prefacs_A0()
elif FF == 'A1':
    mom,pref=GetCombs.get_moms_and_prefacs_A1()
elif FF == 'A2':
    mom,pref=GetCombs.get_moms_and_prefacs_A2()
   

# Number of combinations of directions
nmom=len(mom[nsq])


# Read 3pt data
dsets=[f3pt["/CHARM_PT_SEQ_SM{}_s{}/c{}/dT{}/{}/forward/data".format(sm,smass,cmass,dt,mom[nsq][i])] for i in range(len(mom[nsq]))]
dsetsb=[f3pt["/CHARM_PT_SEQ_SM{}_s{}/c{}/dT{}/{}/backward/data".format(sm,smass,cmass,dt,mom[nsq][i])] for i in range(len(mom[nsq]))]


# Read 2pt data
'''
dsxn0=f2ptDs["/cl_SM{}_SM{}_{}/c{}/operator_GammaX/n2_{}/data".format(sm,sm,smass,cmass,nsq)]
dsyn0=f2ptDs["/cl_SM{}_SM{}_{}/c{}/operator_GammaY/n2_{}/data".format(sm,sm,smass,cmass,nsq)]
dszn0=f2ptDs["/cl_SM{}_SM{}_{}/c{}/operator_GammaZ/n2_{}/data".format(sm,sm,smass,cmass,nsq)]
bsn0=f2ptBs["/hl_SM{}_SM{}_{}_m{}_csw{}_zeta{}/operator_Gamma5/n2_0/data".format(sm,sm,smass,m,csw,zeta)]
'''

#For old F1S format
ptmom=['0_0_0','1_0_0','1_1_0','1_1_1', '2_0_0','2_1_0']
dsxn0=f2ptDs["/CHARM_SM{}_SM{}_s{}/c{}/operator_GammaX/{}/data".format(sm,sm,smass,cmass,ptmom[nsq])]
dsyn0=f2ptDs["/CHARM_SM{}_SM{}_s{}/c{}/operator_GammaY/{}/data".format(sm,sm,smass,cmass,ptmom[nsq])]
dszn0=f2ptDs["/CHARM_SM{}_SM{}_s{}/c{}/operator_GammaZ/{}/data".format(sm,sm,smass,cmass,ptmom[nsq])]
bsn0=f2ptBs["/rhq_m2.42_csw2.68_zeta1.52_SM{}_SM{}_s{}/operator_Gamma5/0_0_0/data".format(sm,sm,smass)]

# Additional datasets for A2
if FF == 'A2':
    A0comp = np.load('../Results/{}/Ratios/A0/Jackknife/nsq{}.npy'.format(ensemble,nsq))
    A1comp = np.load('../Results/{}/Ratios/A1/Jackknife/nsq{}.npy'.format(ensemble,nsq))
    A0fit=np.load('../Results/{}/Fits/A0/A0-nsq{}.npy'.format(ensemble, nsq))
    A1fit=np.load('../Results/{}/Fits/A1/A1-nsq{}.npy'.format(ensemble, nsq))

# Read eff. fit jackknife blocks
bsfit=pd.read_csv('../Data/{}/2pt/Bs-blocks.csv'.format(ensemble),sep='\s')
dsfit=pd.read_csv('../Data/{}/2pt/Ds{}-nsq{}-blocks.csv'.format(ensemble,cmass,nsq),sep='\s')


# Prefactor and folding
if FF == 'V':
    pre=-(mb+md)/(2*mb)*L/(2*np.pi)
    av1n0=Folding.folding3ptVec(dsets, dsetsb, nmom, dt, nconf, ts)
elif FF == 'A0':
    pre=md / (2 * ed * mb)
    av1n0=Folding.folding3ptAx(dsets, dsetsb, nmom, dt, nconf, ts)   
elif FF == 'A1':
    pre=1/(mb+ed)
    av1n0=Folding.folding3ptAx(dsets, dsetsb, nmom, dt, nconf, ts)   
elif FF == 'A2':    
    #pre=(mb+md)*md**2*(mb-ed)/(2*(mb**2*ed))
    #pre=md**2*(mb-md)/(ed*mb)*(ed*mb+md**2)/(ed*mb-mb**2)*(ed**2-2*md**2+ed*mb)/(md**2-ed*mb)
    pre=md**2*(mb+md)/(ed*mb)
    av1n0=Folding.folding3ptAx(dsets, dsetsb, nmom, dt, nconf, ts)   
    #av1n02=Folding.folding3ptAx(dsets2, dsetsb2, nmom, dt, nconf, ts)   
   
avdx,avdy,avdz=Folding.folding2pt3(dsxn0, dsyn0, dszn0, nmom, dt, nconf, ts)
avb=Folding.folding2pt(bsn0, nmom, dt, nconf, ts)


#Create Jackknife Blocks, nconf's component is the mean
jb3pt=Jackblocks.create_blocks_3pt(av1n0,nmom, dt, nconf)
#if FF == 'A2': jb3pt2=Jackblocks.create_blocks_3pt(av1n02,nmom, dt, nconf)
jbdx=Jackblocks.create_blocks_2pt(avdx,dt,nconf)
jbdy=Jackblocks.create_blocks_2pt(avdy,dt,nconf)
jbdz=Jackblocks.create_blocks_2pt(avdz,dt,nconf)
jbb=Jackblocks.create_blocks_2pt(avb,dt,nconf)

# Calculate Ratio
if FF == 'A2':
    ratiojack,errn0 =Ratio.build_Ratio_A2(jb3pt,jbdx,jbdy,jbdz,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,A0comp,A1comp,L,A0fit,A1fit)
else:
    ratiojack,errn0 =Ratio.build_Ratio(jb3pt,jbdx,jbdy,jbdz,jbb,pref,dt,nsq,nconf,ed,mb,pre,dsfit,bsfit)

avn0=ratiojack[:,nconf]
# Save Data and Plot
plt.xlabel('time')
plt.ylabel(rf'$\widetilde{{{FF}}}$')
plt.errorbar(list(range(dt)), np.absolute(avn0)[0:dt], yerr=errn0[0:dt],ls='none',fmt='x',label='nsq={}'.format(nsq))
plt.legend()
plt.savefig('../Results/{}/Ratios/{}/{}-nsq{}.png'.format(ensemble,FF,FF,nsq))
np.savetxt('../Results/{}/Ratios/{}/{}-nsq{}.txt'.format(ensemble,FF,FF,nsq), np.c_[np.absolute(avn0), errn0])
np.save('../Results/{}/Ratios/{}/Jackknife/nsq{}.npy'.format(ensemble,FF,nsq), ratiojack)


###############################################################################


#Covarianze matrix (without prefactor, not squarrooted)
#covmat=Regression.build_Covarianz(reg_up, reg_low, ts, jb3pt, jbdx, jbdy, jbdz, jbb, pref, dt, nsq, nconf, md, mb, pre, dsfit, bsfit, avn0)
covmat=Regression.build_Covarianz_A2(reg_up,reg_low,ts,jb3pt,jbdx,jbdy,jbdz,jbb,pref,dt,nsq,nconf,md,mb,ed,pre,dsfit,bsfit,A0comp,A1comp,L,A0fit,A1fit,avn0)

cut=ts/2-1-reg_up

def chi(a):
    return (nconf-1-reg_low-cut)/(nconf-reg_low-cut)*np.dot(np.transpose([i-a for i in avn0[reg_low:reg_up]]),np.matmul(np.linalg.inv(covmat),[i-a for i in avn0[reg_low:reg_up]]))

mbar=minimize(chi,0.1,method='Nelder-Mead', tol=1e-6)
'''
def jackmass(t1,i):
    return (((Basic.sum_with_prefacs(jb3pt[:,t1,i], pref[nsq],nsq)))/(np.sqrt(1/3*(jbdx[t1,i]+jbdy[t1,i]+jbdz[t1,i])*jbb[dt-(t1),i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*(t1))*np.exp(-bsfit['EffectiveMass'][i]*(dt-(t1)))))*pre

def chijack(a,k):
    return np.dot(np.transpose([jackmass(i+reg_low,k)-a for i in range(int(ts/2-1-reg_low-cut))]),np.matmul(np.linalg.inv(covmat),[jackmass(i+reg_low,k)-a for i in range(int(ts/2-1-reg_low-cut))]))
'''
def chijack(a,k):
    return np.dot(np.transpose([Regression.jackratio_A2(k,i + reg_low,jb3pt,jbdx,jbdy,jbdz,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,A0comp,A1comp,L,A0fit,A1fit) - a for i in range(int(ts / 2 - 1 - reg_low - cut))]),
                  np.matmul(np.linalg.inv(covmat),
                            [Regression.jackratio_A2(k,i + reg_low,jb3pt,jbdx,jbdy,jbdz,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,A0comp,A1comp,L,A0fit,A1fit) - a for i in range(int(ts / 2 - 1 - reg_low - cut))]))


#Std Deviatson for all jakcknife blocks

jblocks=np.zeros(nconf)
h=0
for i in range(nconf):
    tmp=minimize(chijack,0.1,args=(i),method='Nelder-Mead', tol=1e-6).x[0]
    jblocks[i]=tmp
    h=h+(minimize(chijack,0.1,args=(i),method='Nelder-Mead', tol=1e-6).x[0]-mbar.x[0])**2
sigma=np.sqrt((nconf-1-reg_low-cut)/(nconf-reg_low-cut)*h)

#np.savetxt('../Results/{}/Fits/{}/{}-nsq{}.txt'.format(ensemble,FF,FF,nsq), np.c_[np.absolute(avn0), errn0])
np.save('../Results/{}/Fits/{}/{}-nsq{}.npy'.format(ensemble,FF,FF,nsq), jblocks)

#df4 = pd.DataFrame(columns=['EffectiveMass'])
#df4['EffectiveMass']=jblocks
#df4.to_csv('../Results/{}/Fits/{}/{}-nsq{}-Block.csv'.format(ensemble,FF,FF,nsq), sep='\t')

print(mbar,sigma)

f = plt.figure(figsize=(10,3))
ax = f.add_subplot(121)

ax.set_xlabel('time')
ax.set_ylabel(rf'$\widetilde{{{FF}}}$')
ax.errorbar(list(range(dt))[1:dt], avn0[1:dt], yerr=errn0[1:dt],fmt='x', label='nsq={}'.format(nsq))
plt.axhline(y = mbar.x[0], color = 'r', linestyle = 'dashed')
plt.fill_between(list(range(dt))[reg_low:reg_up], mbar.x[0]+sigma, mbar.x[0]-sigma, color='r',alpha=0.2)
#ax.set_yscale('log')

plt.savefig('../Results/{}/Fits/{}/{}-Av-nsq{}-Fit.png'.format(ensemble,FF,FF,nsq))

df3 = pd.DataFrame(columns=['EffectiveMass','Error','RegUp','RegLow'])
df3['EffectiveMass']=mbar.x
df3['Error']=sigma  
df3['RegUp']=reg_up
df3['RegLow']=reg_low    
df3.to_csv('../Results/{}/Fits/{}/{}-Av-nsq{}-Fit.csv'.format(ensemble,FF,FF,nsq), sep='\t')
