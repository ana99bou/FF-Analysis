import numpy as np
import Basic

# Building the ratio and the ratio of jackknife blokcs

def build_Ratio(jb3pt,jbdx,jbb,pref,dt,nsq,nconf,ed,mb,pre,dsfit,bsfit):
    avn0 = np.zeros(dt+1)
    errn0=np.zeros(shape=(dt+1))
    ratiojack=np.zeros(shape=(dt+1,nconf+1))
    for j in range(dt+1):   
        ratiojack[j][nconf]=pre * (jb3pt[j,nconf]/ ( np.sqrt(jbdx[j,nconf] * jbb[dt-j,nconf]))) * np.sqrt((4 * mb * ed) / (np.exp(-ed * j) * np.exp(-mb * (dt - j))))
        
        x=0
        for i in range(nconf):
            #x=x+((Basic.sum_with_prefacs(jb3pt[:,j,i], pref[nsq],nsq)/(np.sqrt(1/3*(jbdx[j,i]+jbdy[j,i]+jbdz[j,i])*jbb[dt-j,i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*j)*np.exp(-bsfit['EffectiveMass'][i]*(dt-j))))*pre-avn0[j])**2
            ratiojack[j][i]=(jb3pt[j,i]/(np.sqrt(jbdx[j,i]*jbb[dt-j,i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*j)*np.exp(-bsfit['EffectiveMass'][i]*(dt-j))))*pre
            x=x+(ratiojack[j][i]-ratiojack[j][nconf])**2
        errn0[j]=np.sqrt((nconf-1)/nconf*x) 
    avn0[np.isnan(avn0)] = 0
    errn0[np.isnan(errn0)] = 0
    return ratiojack,errn0


##Need ot build ratio over one comp and then average in this case, num gives the number in the list

def build_Ratio_A2(jb3pt,jbdx,jbb,pref,dt,nsq,nconf,pre,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit):
    avn0 = np.zeros(dt+1)
    errn0=np.zeros(shape=(dt+1))
    ratiojack=np.zeros(shape=(dt+1,nconf+1))
    print(A0fit)
    print(A1fit)
    for j in range(dt+1):
        #print(build_A2(nconf,j,jb3pt,jbdx,jbdy,jbdz,jbb,pref,dt,nsq,nconf,md,mb,ed,dsfit,bsfit,A0comp[j,nconf],A1comp[j,nconf]))
        ratiojack[j][nconf]=pre*build_A2(nconf,j,jb3pt,jbdx,jbb,pref,dt,nsq,nconf,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit)
        #avn0[j] = pre * (Basic.sum_with_prefacs(jb3pt[:,j,nconf],pref[nsq], nsq) / ( np.sqrt(1/3*(jbdx[j,nconf] + jbdy[j,nconf] + jbdz[j,nconf]) * jbb[dt-j,nconf]))) * np.sqrt((4 * mb * md) / (np.exp(-md * j) * np.exp(-mb * (dt - j))))
        x=0
        for i in range(nconf):
            ratiojack[j][i]=pre*build_A2(i,j,jb3pt,jbdx,jbb,pref,dt,nsq,i,md,mb,ed,dsfit,bsfit,L,A0fit,A1fit)
            x=x+(ratiojack[j][i]-ratiojack[j][nconf])**2
            #print(i,j)
            #print(ratiojack[j][i]-ratiojack[j][nconf])
        errn0[j]=np.sqrt((nconf-1)/nconf*x) 
    avn0[np.isnan(avn0)] = 0
    errn0[np.isnan(errn0)] = 0
    return ratiojack,errn0


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
    #qsq=mb**2-md**2+2*ed**2-2*md*ed
    #qsq=(mb-ed)**2+nsq
    #L=48
    conv=(2*np.pi/L)
    #conv=1
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

'''
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

#alles passiert in basic sum
def build_Ratio_A2_new(jb3pt,jbdx,jbdy,jbdz,jbb,pref,dt,nsq,nconf,md,mb,ed,pre,dsfit,bsfit,A0comp,A1comp):
    avn0 = np.zeros(dt+1)
    errn0=np.zeros(shape=(dt+1))
    for j in range(dt+1):    
        avn0[j]=pre*(Basic.sum_with_prefacs_A2(jb3pt[:,j,nconf], pref, A0comp, A1comp, mb, md, ed, nsq))/(np.sqrt(1/3*(jbdx[j,nconf]+jbdy[j,nconf]+jbdz[j,nconf])*jbb[dt-j,nconf]))*np.sqrt((4*mb*md)/(np.exp(-md*j)*np.exp(-mb*(dt-j))))
        x=0
        for i in range(nconf):
           x=x+(pre*(Basic.sum_with_prefacs(jb3pt2[:,j,i],pref2[nsq], nsq)+Basic.sum_with_prefacs(jb3pt[:,j,nconf],pref[nsq], nsq))/(np.sqrt(1/3*(jbdx[j,i]+jbdy[j,i]+jbdz[j,i])*jbb[dt-j,i]))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*j)*np.exp(-bsfit['EffectiveMass'][i]*(dt-j))))-avn0[j])**2
           #x=x+(pre*(Basic.sum_with_prefacs_A2(jb3pt[:,j,i],pref[nsq],jb3pt2[:,j,nconf],pref2[nsq], nsq))/(np.sqrt(1/3*(jbdx[j,i]+jbdy[j,i]+jbdz[j,i])*jbb[dt-j,i]))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*j)*np.exp(-bsfit['EffectiveMass'][i]*(dt-j))))-avn0[j])**2
        errn0[j]=np.sqrt((nconf-1)/nconf*x) 
    avn0[np.isnan(avn0)] = 0
    errn0[np.isnan(errn0)] = 0
    return avn0,errn0




def build_Covarianz(reg_up,reg_low,ts,nconf):
    cut=ts/2-1-reg_up
    cut1=ts/2+1-reg_up
    covmat=np.zeros(shape=(int(ts/2-1-reg_low-cut),int(ts/2-1-reg_low-cut)))
    for t1 in range(int(ts/2-1-reg_low-cut)):
        for t2 in range(int(ts/2-1-reg_low-cut)):
            x=0
            for i in range(nconf):  
                #x=x+(Basic.sum_with_prefacs(jb3pt[:,t1+reg_low,i], pref[nsq],nsq)/(np.sqrt(1/3*(jbdx[t1+reg_low,i]+jbdy[t1+reg_low,i]+jbdz[t1+reg_low,i])*jbb[dt-t1+reg_low,i])))*np.sqrt((4*dsfit['EffectiveMass'][i]*bsfit['EffectiveMass'][i])/(np.exp(-dsfit['EffectiveMass'][i]*j)*np.exp(-bsfit['EffectiveMass'][i]*(dt-(t1+reg_low)))))*pre
            covmat[t1][t2]=x
            covmat[t2][t1]=x  
'''        