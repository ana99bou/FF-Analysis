import numpy as np
import Basic


def folding3ptVec(dsets,dsetsb,nmom,dt,nconf,ts,pref,nsq):
    
    # Initialize arrays
    av1n0=np.zeros((dt+1,nconf))
    for j in range(dt+1):
        for k in range(nconf):
            tmp=[np.mean((np.imag(dsets[i][k, :, j]) - np.imag(dsetsb[i][k, :, dt-j]))) / 2 for i in range(nmom)]
            for l in range(nmom):
                av1n0[j,k]=Basic.sum_with_prefacs(pref[nsq],tmp[:],1)
    return av1n0


def folding3ptAx(dsets,dsetsb,nmom,dt,nconf,ts,pref,nsq):
    # Initialize arrays
    av1n0=np.zeros((dt+1,nconf))
    for j in range(dt+1):
        for k in range(nconf):
            tmp=[np.mean((np.real(dsets[i][k, :, j]) + np.real(dsetsb[i][k, :, dt-j]))) / 2 for i in range(nmom)]
            for l in range(nmom):
                av1n0[j,k]=Basic.sum_with_prefacs(pref[nsq],tmp[:],1)
    return av1n0


def folding3ptAxA2(dsets,dsetsb,nmom,dt,nconf,ts,pref,nsq):
    # Initialize arrays
     av1n0=np.zeros((nmom, dt+1,nconf))
     for j in range(dt+1):
         for k in range(nconf):
             tmp=[np.mean((np.real(dsets[i][k, :, j]) + np.real(dsetsb[i][k, :, dt-j]))) / 2 for i in range(nmom)]
             for l in range(nmom):
                 av1n0[l][j,k]=tmp[l]
     return av1n0


def folding2pt(dsxn0,nmom,dt,nconf,ts):
    avdx = np.zeros((dt+1, nconf))

    for j in range(dt+1):  
        for k in range(nconf):
            tmpdx = np.mean(np.real(dsxn0[k, :, j]) + np.real(dsxn0[k, :, ts-j])) / 2 if j != 0 else np.mean(np.real(dsxn0[k, :, 0]))
            avdx[j, k] = tmpdx
    return avdx
        


#For all 3 directions at once
'''
def folding2pt3(dsxn0,dsyn0,dszn0,nmom,dt,nconf,ts):
    avdx = np.zeros((dt+1, nconf))
    avdy = np.zeros((dt+1, nconf))
    avdz = np.zeros((dt+1, nconf))
    
    for j in range(dt+1):  
        for k in range(nconf):
            tmpdx = np.mean(np.real(dsxn0[k, :, j]) + np.real(dsxn0[k, :, ts-j])) / 2 if j != 0 else np.mean(np.real(dsxn0[k, :, 0]))
            tmpdy = np.mean(np.real(dsyn0[k, :, j]) + np.real(dsyn0[k, :, ts-j])) / 2 if j != 0 else np.mean(np.real(dsyn0[k, :, 0]))
            tmpdz = np.mean(np.real(dszn0[k, :, j]) + np.real(dszn0[k, :, ts-j])) / 2 if j != 0 else np.mean(np.real(dszn0[k, :, 0]))
        
            avdx[j, k] = tmpdx
            avdy[j, k] = tmpdy
            avdz[j, k] = tmpdz
            
    return avdx,avdy,avdz
'''


def folding2pt3(dsxn0,dsyn0,dszn0,nmom,dt,nconf,ts):
    avdx = np.zeros((dt+1, nconf))
    avdy = np.zeros((dt+1, nconf))
    avdz = np.zeros((dt+1, nconf))
    
    for j in range(dt+1):  
        for k in range(nconf):
            tmpdx = (np.mean(np.real(dsxn0[k, :, j]) + np.real(dsxn0[k, :, ts-j])) / 2 + np.mean(np.real(dsyn0[k, :, j]) + np.real(dsyn0[k, :, ts-j])) / 2 + np.mean(np.real(dszn0[k, :, j]) + np.real(dszn0[k, :, ts-j])) / 2)/3 if j != 0 else (np.mean(np.real(dsxn0[k, :, 0]))+np.mean(np.real(dsyn0[k, :, 0]))+np.mean(np.real(dszn0[k, :, 0])))/3
            #tmpdy = np.mean(np.real(dsyn0[k, :, j]) + np.real(dsyn0[k, :, ts-j])) / 2 if j != 0 else np.mean(np.real(dsyn0[k, :, 0]))
            #tmpdz = np.mean(np.real(dszn0[k, :, j]) + np.real(dszn0[k, :, ts-j])) / 2 if j != 0 else np.mean(np.real(dszn0[k, :, 0]))
        
            avdx[j, k] = tmpdx
            #avdy[j, k] = tmpdy
            #avdz[j, k] = tmpdz
            
    return avdx
        