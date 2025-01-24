import numpy as np

def jack(x,j):
    r=0
    for i in range(len(x)):
        if i!=j:
            r=r+x[i]
    return 1/(len(x)-1)*r      

def create_blocks_3pt(dat,nmom,dt,nconf):
    res=np.zeros((nmom,dt+1,nconf+1))
    for mom in range(nmom):
        for t in range(dt+1):
            for i in range(nconf):
                res[mom,t,i]=jack(dat[mom,t],i)
            res[mom,t,nconf]=np.mean(dat[mom,t])
    return res

def create_blocks_2pt(dat,dt,nconf):
    res=np.zeros((dt+1,nconf+1))
    for t in range(dt+1):
        for i in range(nconf):
            res[t,i]=jack(dat[t],i)
        res[t,nconf]=np.mean(dat[t])
    return res