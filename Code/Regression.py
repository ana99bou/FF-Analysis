import numpy as np
import Basic
from scipy.optimize import minimize
import scipy

def pvalue(chi2, dof):
    r"""Compute the $p$-value corresponding to a $\chi^2$ with `dof` degrees
    of freedom."""
    return 1 - scipy.stats.chi2.cdf(chi2, dof)


def build_Covarianz(reg_up,reg_low,data,nconf):
    covmat=np.zeros(shape=(int(reg_up-reg_low),int(reg_up-reg_low)))
    for t1 in range(int(reg_up-reg_low)):
        for t2 in range(int(reg_up-reg_low)):
            x=0
            for i in range(nconf): 
                x=x+(data[t1+reg_low,i]-data[t1+reg_low,nconf])*(data[t2+reg_low,i]-data[t2+reg_low,nconf])
            covmat[t1][t2]=(nconf-1)/nconf*x
            covmat[t2][t1]=(nconf-1)/nconf*x  
    return covmat

def get_fit(reg_up,reg_low,covmat,nconf,data,fit_function):
    invcovmat=np.linalg.inv(covmat)

    #k=nconf for mean
    #insert function in terms of a
    def build_chisquare(a,k):
        return np.dot(np.transpose([i-fit_function(a) for i in data[reg_low:reg_up,k]]),np.matmul(invcovmat,[i-fit_function(a) for i in data[reg_low:reg_up,k]]))
    fit_res = minimize(build_chisquare, 0.1, args=(nconf), method='Nelder-Mead', tol=1e-8)
    pval=pvalue(fit_res.fun, reg_up-reg_low)

    #uncertainty
    jblocks=np.zeros(nconf)
    h=0
    for i in range(nconf):
        tmp=minimize(build_chisquare,0.1,args=(i),method='Nelder-Mead', tol=1e-8).x[0]
        jblocks[i]=tmp
        h=h+(tmp-fit_res.x[0])**2
    sigma=np.sqrt((nconf-1)/(nconf)*h)

    return fit_res.x[0], sigma, pval



########OLD

def build_Covarianz_OLD(reg_up,reg_low,ts,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,md,mb,pre,dsfit,bsfit,avn0):
    cut=ts/2-1-reg_up
    cut1=ts/2+1-reg_up
    covmat=np.zeros(shape=(int(ts/2-1-reg_low-cut),int(ts/2-1-reg_low-cut)))
    for t1 in range(int(ts/2-1-reg_low-cut)):
        for t2 in range(int(ts/2-1-reg_low-cut)):
            x=0
            for i in range(nconf): 
                #x=x+(((Basic.sum_with_prefacs(jb3pt[:,t1+reg_low,i], pref[nsq],nsq))/(np.sqrt(1/3*(jbdx[t1+reg_low,i]+jbdy[t1+reg_low,i]+jbdz[t1+reg_low,i])*jbb[dt-(t1+reg_low),i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*(t1+reg_low))*np.exp(-bsfit['EffectiveMass'][i]*(dt-(t1+reg_low)))))*pre-avn0[t1])*(((Basic.sum_with_prefacs(jb3pt[:,t2+reg_low,i], pref[nsq],nsq))/(np.sqrt(1/3*(jbdx[t2+reg_low,i]+jbdy[t2+reg_low,i]+jbdz[t2+reg_low,i])*jbb[dt-(t2+reg_low),i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*(t2+reg_low))*np.exp(-bsfit['EffectiveMass'][i]*(dt-(t2+reg_low)))))*pre-avn0[t2])
                x=x+(jackratio(t1, i, reg_up, reg_low, ts, jb3pt, jbdx, jbb, pref, dt, nsq, nconf, md, mb, pre, dsfit, bsfit, avn0)-avn0[t1])*(jackratio(t2, i, reg_up, reg_low, ts, jb3pt, jbdx, jbb, pref, dt, nsq, nconf, md, mb, pre, dsfit, bsfit, avn0)-avn0[t2])
            covmat[t1][t2]=x
            covmat[t2][t1]=x  
    return covmat

def chi(a,reg_up,reg_low,nconf,cut,avn0,covmat):
    return (nconf-1-reg_low-cut)/(nconf-reg_low-cut)*np.dot(np.transpose([i-a for i in avn0[reg_low:reg_up]]),np.matmul(np.linalg.inv(covmat),[i-a for i in avn0[reg_low:reg_up]]))

def jackratio(t1,i,reg_up,reg_low,ts,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,md,mb,pre,dsfit,bsfit,avn0):
    return (((jb3pt[t1,i]))/(np.sqrt(jbdx[t1,i]*jbb[dt-(t1),i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*(t1))*np.exp(-bsfit['EffectiveMass'][i]*(dt-(t1)))))*pre

def chijack(a,k,ts,reg_up,reg_low,nconf,cut,avn0,covmat):
    return np.dot(np.transpose([jackratio(i+reg_low,k)-a for i in range(int(ts/2-1-reg_low-cut))]),np.matmul(np.linalg.inv(covmat),[jackratio(i+reg_low,k)-a for i in range(int(ts/2-1-reg_low-cut))]))

def fitting(nconf,mbar,reg_low,reg_up,cut):
    h=0
    mbar=minimize(chi,0.1,method='Nelder-Mead', tol=1e-6)
    for i in range(nconf):
        h=h+(minimize(chijack,0.1,args=(i),method='Nelder-Mead', tol=1e-6).x[0]-mbar.x[0])**2
    sigma=np.sqrt((nconf-1-reg_low-cut)/(nconf-reg_low-cut)*h)
    return mbar,sigma

###################################

def build_Covarianz_A2(reg_up,reg_low,ts,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,md,mb,ed,pre,dsfit,bsfit,L,A0fit,A1fit,avn0):
    cut=ts/2-1-reg_up
    cut1=ts/2+1-reg_up
    covmat=np.zeros(shape=(int(ts/2-1-reg_low-cut),int(ts/2-1-reg_low-cut)))
    for t1 in range(int(ts/2-1-reg_low-cut)):
        for t2 in range(int(ts/2-1-reg_low-cut)):
            x=0
            for i in range(nconf):
                #x=x+(((Basic.sum_with_prefacs(jb3pt[:,t1+reg_low,i], pref[nsq],nsq))/(np.sqrt(1/3*(jbdx[t1+reg_low,i]+jbdy[t1+reg_low,i]+jbdz[t1+reg_low,i])*jbb[dt-(t1+reg_low),i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*(t1+reg_low))*np.exp(-bsfit['EffectiveMass'][i]*(dt-(t1+reg_low)))))*pre-avn0[t1])*(((Basic.sum_with_prefacs(jb3pt[:,t2+reg_low,i], pref[nsq],nsq))/(np.sqrt(1/3*(jbdx[t2+reg_low,i]+jbdy[t2+reg_low,i]+jbdz[t2+reg_low,i])*jbb[dt-(t2+reg_low),i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*(t2+reg_low))*np.exp(-bsfit['EffectiveMass'][i]*(dt-(t2+reg_low)))))*pre-avn0[t2])
                x=x+(jackratio_A2(i,t1,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit)-avn0[t1])*(jackratio_A2(i,t2,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit)-avn0[t2])
            covmat[t1][t2]=x
            covmat[t2][t1]=x
    return covmat


def jackratio_A2(i,j,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit):
    return pre*build_A2(i,j,jb3pt,jbdx,jbb,pref,dt,nsq,i,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit)

def chijack(a,k,ts,reg_up,reg_low,nconf,cut,avn0,covmat):
    return np.dot(np.transpose([jackratio_A2(i+reg_low,k)-a for i in range(int(ts/2-1-reg_low-cut))]),np.matmul(np.linalg.inv(covmat),[jackratio(i+reg_low,k)-a for i in range(int(ts/2-1-reg_low-cut))]))


def build_A2(i,j,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit):
    total=0
    pref=pref[nsq]
    #A0tmp=0.3275769042968758
    #A1tmp=0.4757202148437489
    #A0tmp=0.3172631835937508
    #A1tmp=0.46430969238281383
    #A0tmp=0.30290161132812576
    #A1tmp=0.44184814453125126
    #A0tmp=0.29032226562500074
    #A1tmp=0.43177917480468875
    conv=(2*np.pi/L)
    qsq=mb**2+md**2-2*mb*ed
    #qsq=(mb-ed)**2+(2*np.pi/L)**2*nsq
    for num in range(len(pref)):
        #total+=1/(pref[num][1]**2*conv**2)*1/(1+(md**2-mb**2)/qsq)*(-2*(pref[num][1]**2*conv**2)*ed*mb/(qsq*md)*A0comp+(mb+md)*(1+(pref[num][1]**2*conv**2)/md**2+(ed*mb*(pref[num][1]**2*conv**2))/(md**2*qsq))*A1comp-pref[num][0]*build_mat(num,j,i,jb3pt,jbdx,jbdy,jbdz,jbb,pref,dt,nsq,nconf,md,mb,ed,dsfit,bsfit))
        #print(-2*ed*mb/(qsq*md)*A0comp+(mb+md)*(1/md**2+(ed*mb)/(md**2*qsq))*A1comp+(mb+md)/(pref[num][1]**2*conv**2)*A1comp,-1/(pref[num][1]**2*conv**2)*pref[num][0]*build_mat(num,j,i,jb3pt,jbdx,jbdy,jbdz,jbb,pref,dt,nsq,nconf,md,mb,ed,dsfit,bsfit))
        #total+=qsq/(qsq+mb**2-md**2)*(-2*ed*mb/(qsq*md)*A0comp+(mb+md)*(1/md**2+(ed*mb)/(md**2*qsq))*A1comp+(mb+md)/(pref[num][1]**2*conv**2)*A1comp-1/(pref[num][1]**2*conv**2)*pref[num][0]*build_mat(num,j,i,jb3pt,jbdx,jbdy,jbdz,jbb,pref,dt,nsq,nconf,md,mb,ed,dsfit,bsfit))
        total += qsq / (qsq + mb ** 2 - md ** 2) * (-2 * ed * mb / (qsq * md) * A0fit+ (mb + md) * (
                    1 / md ** 2 + (ed * mb) / (md ** 2 * qsq)) * A1fit + (mb + md) / (
                                                                pref[num][1] ** 2 * conv ** 2) * A1fit - 1 / (
                                                                pref[num][1] ** 2 * conv ** 2) * pref[num][
                                                        0] * build_mat(num, j, i, jb3pt, jbdx, jbb, pref,
                                                                       dt, nsq, nconf, md, mb, ed, dsfit, bsfit))
    return total/len(pref)

def build_mat(num,j,i,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,md,mb,ed,dsfit,bsfit):
    if i == nconf: return (jb3pt[num,j,i]/(np.sqrt(jbdx[j,i]*jbb[dt-j,i])))*np.sqrt((4*ed*mb)/(np.exp(-ed*j)*np.exp(-mb*(dt-j))))
    else: return (jb3pt[num,j,i]/(np.sqrt(jbdx[j,i]*jbb[dt-j,i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*j)*np.exp(-bsfit['EffectiveMass'][i]*(dt-j))))


