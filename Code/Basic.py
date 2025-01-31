import numpy as np
import scipy


def pvalue(chi2, dof):
    r"""Compute the $p$-value corresponding to a $\chi^2$ with `dof` degrees
    of freedom."""
    return 1 - scipy.stats.chi2.cdf(chi2, dof)

def jack(x,j):
    r=0
    for i in range(len(x)):
        if i!=j:
            r=r+x[i]
    return 1/(len(x)-1)*r        

def exp_val(data):
    return sum(data)/len(data)

def var(data):
    res=0
    for i in range(len(data)):
        res=res+(jack(data,i)-exp_val(data))**2
    return np.sqrt((len(data)-1)/len(data)*res)

def jack_mean(dat, nmom,j,i):
    temp=0
    for k in range(nmom):
        temp += jack(dat[k][j],i)
    return temp/nmom

def sum_with_prefacs(lst,pref,nsq):
    total=0
    for i in range(0,len(lst)):
        total+=pref[i]*lst[i]
    return total/len(lst)

#include k^2 information as second input in pref list in terms of pref [i,1]
def sum_with_prefacs_A2(lst,pref,A0comp,A1comp,mb,md,ed,nsq):
    total=0
    qsq=mb**2-md**2+2*ed**2-2*md*ed
    for i in range(0,len(lst)):
        total+=1/pref[i][1]**2*1/(1+(md**2-mb**2)/qsq)*(-2*pref[i][1]**2*ed*mb/(qsq*md)*A0comp+(mb+md)*(1+pref[i][1]**2/md**2+(ed*mb*pref[i][1]**2)/(md**2*qsq))*A1comp-pref[i][0]*lst[i])
    return total/len(lst)
        
def sum_with_prefacs_jack(lst,pref,nsq,j,i):
    total=0
    for k in range(0,len(lst)):
        #total+=V_GetCombs.getPrefacs()[k]*jack(lst[k][j],i)
        total+=pref[k]*jack(lst[k][j],i)
    return total/len(lst)

def sum_with_prefacs_jack_A2(lst,lst2,pref,pref2,nsq,j,i):
    total=0
    for k in range(0,len(lst)):
        #total+=V_GetCombs.getPrefacs()[k]*jack(lst[k][j],i)
        total+=pref[k]*jack(lst[k][j],i)+pref2[k]*jack(lst2[k][j],i)
    return total/len(lst)

#information on i and j included in block
#def sum_with_prefacs_jack(block,pref,nsq):
#    total=0
#    for k in range(0,len(block)):
#        #total+=V_GetCombs.getPrefacs()[k]*jack(lst[k][j],i)
#        total+=pref[k]*block[k]
#    return total/len(block)
