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
import scipy

def pvalue(chi2, dof):
    r"""Compute the $p$-value corresponding to a $\chi^2$ with `dof` degrees
    of freedom."""
    return 1 - scipy.stats.chi2.cdf(chi2, dof)
 
#########Choose Params
#######NEED TO RERUN F1S ,CMASS2
FF='V'
nsq=1
ensemble='F1S'
cmass=Ens.getCmass(ensemble)[0] #Ens.getCmass(ensemble) gives us an array of the different charm masses for each ens; chose which one

#C1 12-16 laufen lassen, F1S 18-25
#M2 19-24
reg_up=26
reg_low=20
##########

# Get strange mass and smearing radius
sm=Ens.getSM(ensemble)
smass=Ens.getSmass(ensemble)


# Ensemble details
nconf,dt,ts,L= Ens.getEns(ensemble)
# Not needed for old F1S
#if ensemble != 'F1S':
m,csw,zeta=Ens.getRHQparams(ensemble)

# Read h5 file -> 
##################TODO three h5files for each 3pt and 2pt and get the naming consistent

'''
if ensemble == 'F1S':
    f3pt = h5py.File("../Data/{}/BsDsStar.h5".format(ensemble,ensemble), "r")
    f2ptDs = h5py.File("../Data/{}/BsDsStar.h5".format(ensemble,ensemble), "r")
    f2ptBs = h5py.File("../Data/{}/BsDsStar.h5".format(ensemble,ensemble), "r")
else:
'''
f3pt = h5py.File("../Data/{}/BsDsStar_{}_3pt.h5".format(ensemble,ensemble), "r")
f2ptDs = h5py.File("../Data/{}/BsDsStar_{}_2ptDs.h5".format(ensemble,ensemble), "r")
f2ptBs = h5py.File("../Data/{}/BsDsStar_{}_2ptBs.h5".format(ensemble,ensemble), "r")


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
#if ensemble == 'F1S':
#    ptmom=['0_0_0','1_0_0','1_1_0','1_1_1', '2_0_0','2_1_0']
#    dsxn0=f2ptDs["/CHARM_SM{}_SM{}_s{}/c{}/operator_GammaX/{}/data".format(sm,sm,smass,cmass,ptmom[nsq])]
#    dsyn0=f2ptDs["/CHARM_SM{}_SM{}_s{}/c{}/operator_GammaY/{}/data".format(sm,sm,smass,cmass,ptmom[nsq])]
#    dszn0=f2ptDs["/CHARM_SM{}_SM{}_s{}/c{}/operator_GammaZ/{}/data".format(sm,sm,smass,cmass,ptmom[nsq])]
#    bsn0=f2ptBs["/rhq_m2.42_csw2.68_zeta1.52_SM{}_SM{}_s{}/operator_Gamma5/0_0_0/data".format(sm,sm,smass)]
#else:
dsxn0=f2ptDs["/cl_SM{}_SM{}_{}/c{}/operator_GammaX/n2_{}/data".format(sm,sm,smass,cmass,nsq)]
dsyn0=f2ptDs["/cl_SM{}_SM{}_{}/c{}/operator_GammaY/n2_{}/data".format(sm,sm,smass,cmass,nsq)]
dszn0=f2ptDs["/cl_SM{}_SM{}_{}/c{}/operator_GammaZ/n2_{}/data".format(sm,sm,smass,cmass,nsq)]
bsn0=f2ptBs["/hl_SM{}_SM{}_{}_m{}_csw{}_zeta{}/operator_Gamma5/n2_0/data".format(sm,sm,smass,m,csw,zeta)]

# Additional datasets for A2
if FF == 'A2':
    #A0comp = np.load('../Results/{}/Ratios/A0/Jackknife/nsq{}.npy'.format(ensemble,nsq))
    #A1comp = np.load('../Results/{}/Ratios/A1/Jackknife/nsq{}.npy'.format(ensemble,nsq))
    A0fit=pd.read_csv('../Results/{}/{}/Fits/A0/A0-Av-nsq{}-Fit.csv'.format(ensemble,cmass, nsq),sep='\t')['EffectiveMass']
    A1fit=pd.read_csv('../Results/{}/{}/Fits/A1/A1-Av-nsq{}-Fit.csv'.format(ensemble,cmass, nsq),sep='\t')['EffectiveMass']

# Read eff. fit jackknife blocks
bsfit=pd.read_csv('../Data/{}/2pt/Bs-blocks.csv'.format(ensemble),sep='\s')
dsfit=pd.read_csv('../Data/{}/2pt/Ds{}-nsq{}-blocks.csv'.format(ensemble,cmass,nsq),sep='\s')


# Prefactor and folding
if FF == 'V':
    pre=-(mb+md)/(2*mb)*L/(2*np.pi)
    av1n0=Folding.folding3ptVec(dsets, dsetsb, nmom, dt, nconf, ts,pref,nsq)
elif FF == 'A0':
    pre=md / (2 * ed * mb)
    av1n0=Folding.folding3ptAx(dsets, dsetsb, nmom, dt, nconf, ts,pref,nsq)   
elif FF == 'A1':
    pre=1/(mb+ed)
    av1n0=Folding.folding3ptAx(dsets, dsetsb, nmom, dt, nconf, ts,pref,nsq)   
elif FF == 'A2':    
    #pre=(mb+md)*md**2*(mb-ed)/(2*(mb**2*ed))
    #pre=md**2*(mb-md)/(ed*mb)*(ed*mb+md**2)/(ed*mb-mb**2)*(ed**2-2*md**2+ed*mb)/(md**2-ed*mb)
    pre=md**2*(mb+md)/(ed*mb)
    av1n0=Folding.folding3ptAxA2(dsets, dsetsb, nmom, dt, nconf, ts,pref,nsq)   
    #av1n02=Folding.folding3ptAx(dsets2, dsetsb2, nmom, dt, nconf, ts)   


avdx=Folding.folding2pt3(dsxn0, dsyn0, dszn0, nmom, dt, nconf, ts)
avb=Folding.folding2pt(bsn0, nmom, dt, nconf, ts)

if FF == 'A2':
    jb3pt=Jackblocks.create_blocks_3pt(av1n0,nmom, dt, nconf)
    jbdx=Jackblocks.create_blocks_2pt(avdx,dt,nconf)
    jbb=Jackblocks.create_blocks_2pt(avb,dt,nconf)
else:
    #Create Jackknife Blocks, nconf's component is the mean
    jb3pt=Jackblocks.create_blocks_2pt(av1n0, dt, nconf)
    jbdx=Jackblocks.create_blocks_2pt(avdx,dt,nconf)
    jbb=Jackblocks.create_blocks_2pt(avb,dt,nconf)

jb3_notmean=Jackblocks.jack_not_mean(av1n0,dt,nconf)

# Calculate Ratio
if FF == 'A2':
    ratiojack,errn0 =Ratio.build_Ratio_A2(jb3pt,jbdx,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit)
else:
    ratiojack,errn0 =Ratio.build_Ratio(jb3pt,jbdx,jbb,pref,dt,nsq,nconf,ed,mb,pre,dsfit,bsfit)

avn0=ratiojack[:,nconf]
# Save Data and Plot
plt.xlabel('time')
plt.ylabel(rf'$\widetilde{{{FF}}}$')
plt.errorbar(list(range(dt)), np.absolute(avn0)[0:dt], yerr=errn0[0:dt],ls='none',fmt='x',label='nsq={}'.format(nsq))
plt.legend()
#plt.savefig('../Results/{}/{}/Ratios/{}/{}-nsq{}.png'.format(ensemble,cmass,FF,FF,nsq))
#np.savetxt('../Results/{}/{}/Ratios/{}/{}-nsq{}.txt'.format(ensemble,cmass,FF,FF,nsq), np.c_[np.absolute(avn0), errn0])
#np.save('../Results/{}/{}/Ratios/{}/Jackknife/nsq{}.npy'.format(ensemble,cmass,FF,nsq), ratiojack)
print(ratiojack.shape)
###############################################################################

#Covarianze matrix (without prefactor, not squarrooted)
if FF == 'A2':
    covmat=Regression.build_Covarianz_A2(reg_up,reg_low,ts,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,md,mb,ed,pre,dsfit,bsfit,L,A0fit,A1fit,avn0)
else:
    covmat=Regression.build_Covarianz(reg_up, reg_low, ts, jb3pt, jbdx, jbb, pref, dt, nsq, nconf, md, mb, pre, dsfit, bsfit, avn0)

invcovmat=np.linalg.inv(covmat)

cut=ts/2-1-reg_up
def chi(a):
    return (nconf)/(nconf-1)*np.dot(np.transpose([i-a for i in avn0[reg_low:reg_up]]),np.matmul(invcovmat,[i-a for i in avn0[reg_low:reg_up]]))
#print(avn0[reg_low:reg_up].shape,invcovmat.shape)
mbar=minimize(chi,0.1,method='Nelder-Mead', tol=1e-8)
print(mbar.x[0])

def jackmass(t1,i):
    return (((jb3pt[t1,i]))/(np.sqrt(jbdx[t1,i]*jbb[dt-(t1),i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*(t1))*np.exp(-bsfit['EffectiveMass'][i]*(dt-(t1)))))*pre


if FF == 'A2':
    def chijack(a,k):
        return np.dot(np.transpose([Regression.jackratio_A2(k,i + reg_low,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit) - a for i in range(int(ts / 2 - 1 - reg_low - cut))]),
                  np.matmul(invcovmat,
                            [Regression.jackratio_A2(k,i + reg_low,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit) - a for i in range(int(ts / 2 - 1 - reg_low - cut))]))
else:
    def chijack(a,k):
        return np.dot(np.transpose([jackmass(i+reg_low,k)-a for i in range(int(ts/2-1-reg_low-cut))]),np.matmul(invcovmat,[jackmass(i+reg_low,k)-a for i in range(int(ts/2-1-reg_low-cut))]))


#Std Deviatson for all jakcknife blocks

jblocks=np.zeros(nconf)
h=0
for i in range(nconf):
    tmp=minimize(chijack,0.1,args=(i),method='Nelder-Mead', tol=1e-8).x[0]
    jblocks[i]=tmp
    h=h+(tmp-mbar.x[0])**2
sigma=np.sqrt((nconf-1)/(nconf)*h)

print('Frozen: ',mbar.x[0],sigma)
print('Frozen p-val: ',pvalue(mbar.fun, reg_up-reg_low))


########Unfrozen

unfrozen_tmp=np.zeros((nconf))
mean_unfrozen=0
chi_unfrozen=np.zeros(nconf)
sigma_unfrozen=0
for i in range(nconf):
    covmat_unfrozen=Regression.build_Covarianz(reg_up, reg_low, ts, jb3_notmean[:,i,:], jbdx, jbb, pref, dt, nsq, nconf-1, md, mb, pre, dsfit, bsfit, jb3pt[:,i])
    invcovmat_unfrozen=np.linalg.inv(covmat_unfrozen)
    def chi_unfrozen(a):
        return (nconf-1)/(nconf-2)*np.dot(np.transpose([j-a for j in ratiojack[:,i][reg_low:reg_up]]),np.matmul(invcovmat_unfrozen,[j-a for j in ratiojack[:,i][reg_low:reg_up]]))
    unfrozen_tmp[i]=minimize(chi_unfrozen,0.1,method='Nelder-Mead', tol=1e-8).x[0]
    mean_unfrozen+=unfrozen_tmp[i]
    print(unfrozen_tmp[i])

    def jackmass(t1,l):
        return (((jb3_notmean[:,i,:][t1,l]))/(np.sqrt(jbdx[t1,l]*jbb[dt-(t1),l])))*np.sqrt((4*dsfit['EffectiveMass'][l]*bsfit['EffectiveMass'][l])/(np.exp(-dsfit['EffectiveMass'][l]*(t1))*np.exp(-bsfit['EffectiveMass'][l]*(dt-(t1)))))*pre
    def chijack(a,k):
        return np.dot(np.transpose([jackmass(j+reg_low,k)-a for j in range(int(ts/2-1-reg_low-cut))]),np.matmul(invcovmat_unfrozen,[jackmass(j+reg_low,k)-a for j in range(int(ts/2-1-reg_low-cut))]))
    jblocks_unfrozen=np.zeros(nconf)
    h=0
    for j in range(nconf-1):
        tmp=minimize(chijack,0.1,args=(j),method='Nelder-Mead', tol=1e-8).x[0]
        jblocks_unfrozen[j]=tmp
        #h=h+(tmp-unfrozen_tmp[i])**2
        h=h+(tmp-unfrozen_tmp[i])**2

        #print(tmp-unfrozen_tmp[i])
    print(np.sqrt((nconf-2)/(nconf-1)*h))
    sigma_unfrozen=sigma_unfrozen+np.sqrt((nconf-1)/(nconf)*h)
sigma_unfrozen=sigma_unfrozen/nconf
mean_unfrozen=mean_unfrozen/nconf

print('Unfrozen: ',mean_unfrozen, sigma_unfrozen)

#np.savetxt('../Results/{}/Fits/{}/{}-nsq{}.txt'.format(ensemble,FF,FF,nsq), np.c_[np.absolute(avn0), errn0])
#np.save('../Results/{}/Fits/{}/{}-nsq{}.npy'.format(ensemble,FF,FF,nsq), jblocks)

#df4 = pd.DataFrame(columns=['EffectiveMass'])
#df4['EffectiveMass']=jblocks
#df4.to_csv('../Results/{}/Fits/{}/{}-nsq{}-Block.csv'.format(ensemble,FF,FF,nsq), sep='\t')

f = plt.figure(figsize=(10,3))
ax = f.add_subplot(121)

ax.set_xlabel('time')
ax.set_ylabel(rf'$\widetilde{{{FF}}}$')
ax.errorbar(list(range(dt))[1:dt], avn0[1:dt], yerr=errn0[1:dt],fmt='x', label='nsq={}'.format(nsq))
plt.axhline(y = mbar.x[0], color = 'r', linestyle = 'dashed')
plt.fill_between(list(range(dt))[reg_low:reg_up], mbar.x[0]+sigma, mbar.x[0]-sigma, color='r',alpha=0.2)
#ax.set_yscale('log')

#plt.savefig('../Results/{}/{}/Fits/{}/{}-Av-nsq{}-Fit.png'.format(ensemble,cmass,FF,FF,nsq))

df3 = pd.DataFrame(columns=['EffectiveMass','Error','RegUp','RegLow'])
df3['EffectiveMass']=mean_unfrozen
df3['Error']=sigma_unfrozen
df3['RegUp']=reg_up
df3['RegLow']=reg_low    
df3.to_csv('../Results/{}/{}/Fits/{}/{}-Av-nsq{}-Unfrozen.csv'.format(ensemble,cmass,FF,FF,nsq), sep='\t')

df4 = pd.DataFrame(columns=['pval'])
df4['pval']=pvalue(mbar.fun,reg_up-reg_low)
#df4.to_csv('../Results/{}/{}/Fits/{}/pval-{}-nsq{}.csv'.format(ensemble,cmass,FF,FF,nsq), sep='\t')



