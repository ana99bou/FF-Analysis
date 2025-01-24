import numpy as np
import Basic

# Building the ratio and the ratio of jackknife blokcs

def build_Ratio(jb3pt,jbdx,jbdy,jbdz,jbb,pref,dt,nsq,nconf,md,mb,pre,dsfit,bsfit):
    avn0 = np.zeros(dt+1)
    errn0=np.zeros(shape=(dt+1))
    for j in range(dt+1):    
        avn0[j] = pre * (Basic.sum_with_prefacs(jb3pt[:,j,nconf],pref[nsq], nsq) / ( np.sqrt(1/3*(jbdx[j,nconf] + jbdy[j,nconf] + jbdz[j,nconf]) * jbb[dt-j,nconf]))) * np.sqrt((4 * mb * md) / (np.exp(-md * j) * np.exp(-mb * (dt - j))))
        x=0
        for i in range(nconf):
            x=x+((Basic.sum_with_prefacs(jb3pt[:,j,i], pref[nsq],nsq)/(np.sqrt(1/3*(jbdx[j,i]+jbdy[j,i]+jbdz[j,i])*jbb[dt-j,i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*j)*np.exp(-bsfit['EffectiveMass'][i]*(dt-j))))*pre-avn0[j])**2
        errn0[j]=np.sqrt((nconf-1)/nconf*x) 
    avn0[np.isnan(avn0)] = 0
    errn0[np.isnan(errn0)] = 0
    return avn0,errn0


def build_Ratio_A2(jb3pt,jb3pt2,jbdx,jbdy,jbdz,jbb,pref,pref2,dt,nsq,nconf,md,mb,pre,dsfit,bsfit):
    avn0 = np.zeros(dt+1)
    errn0=np.zeros(shape=(dt+1))
    for j in range(dt+1):    
        avn0[j]=pre*(Basic.sum_with_prefacs(jb3pt2[:,j,nconf],pref2[nsq], nsq)+Basic.sum_with_prefacs(jb3pt[:,j,nconf],pref[nsq], nsq))/(np.sqrt(1/3*(jbdx[j,nconf]+jbdy[j,nconf]+jbdz[j,nconf])*jbb[dt-j,nconf]))*np.sqrt((4*mb*md)/(np.exp(-md*j)*np.exp(-mb*(dt-j))))
        #avn0[j]=pre*(Basic.sum_with_prefacs_A2(jb3pt[:,j,nconf],pref[nsq],jb3pt2[:,j,nconf],pref2[nsq], nsq))/(np.sqrt(1/3*(jbdx[j,nconf]+jbdy[j,nconf]+jbdz[j,nconf])*jbb[dt-j,nconf]))*np.sqrt((4*mb*md)/(np.exp(-md*j)*np.exp(-mb*(dt-j))))
        x=0
        for i in range(nconf):
           x=x+(pre*(Basic.sum_with_prefacs(jb3pt2[:,j,i],pref2[nsq], nsq)+Basic.sum_with_prefacs(jb3pt[:,j,nconf],pref[nsq], nsq))/(np.sqrt(1/3*(jbdx[j,i]+jbdy[j,i]+jbdz[j,i])*jbb[dt-j,i]))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*j)*np.exp(-bsfit['EffectiveMass'][i]*(dt-j))))-avn0[j])**2
           #x=x+(pre*(Basic.sum_with_prefacs_A2(jb3pt[:,j,i],pref[nsq],jb3pt2[:,j,nconf],pref2[nsq], nsq))/(np.sqrt(1/3*(jbdx[j,i]+jbdy[j,i]+jbdz[j,i])*jbb[dt-j,i]))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*j)*np.exp(-bsfit['EffectiveMass'][i]*(dt-j))))-avn0[j])**2
        errn0[j]=np.sqrt((nconf-1)/nconf*x) 
    avn0[np.isnan(avn0)] = 0
    errn0[np.isnan(errn0)] = 0
    return avn0,errn0