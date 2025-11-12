import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ens = 'C2'  # or 'M1', 'M2', etc.
m='0.300'
FF='A1'

bs2ptm0=pd.read_csv('../Data/{}/2pt/Excited-comb-BsResult.csv'.format(ens), sep='\s')['Mass0'].iloc[0]
ds2pt0m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-0.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]
ds2pt1m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-1.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]
ds2pt2m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-2.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]
ds2pt3m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-3.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]
ds2pt4m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-4.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]
ds2pt5m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-5.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]


v1=pd.read_csv('../Results/{}/{}/Fit/Excited-combined-{}.csv'.format(ens,m,FF),comment="#", sep="\t").iloc[:, 1][0]
v2=pd.read_csv('../Results/{}/{}/Fit/Excited-combined-{}.csv'.format(ens,m,FF), comment="#", sep="\t").iloc[:, 1][1]
v3=pd.read_csv('../Results/{}/{}/Fit/Excited-combined-{}.csv'.format(ens,m,FF), comment="#", sep="\t").iloc[:, 1][2]
v4=pd.read_csv('../Results/{}/{}/Fit/Excited-combined-{}.csv'.format(ens,m,FF), comment="#", sep="\t").iloc[:, 1][3]
v5=pd.read_csv('../Results/{}/{}/Fit/Excited-combined-{}.csv'.format(ens,m, FF), comment="#", sep="\t").iloc[:, 1][4]

data=[bs2ptm0,ds2pt0m0,ds2pt1m0, ds2pt2m0,ds2pt3m0,ds2pt4m0,ds2pt5m0,v1,v2,v3,v4,v5]

bs2ptem0=pd.read_csv('../Data/{}/2pt/Excited-comb-BsResult.csv'.format(ens), sep='\s')['DeltaM0'].iloc[0]
ds2pt0em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-0.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]
ds2pt1em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-1.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]
ds2pt2em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-2.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]
ds2pt3em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-3.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]
ds2pt4em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-4.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]
ds2pt5em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-5.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]

v1e=pd.read_csv('../Results/{}/{}/Fit/Excited-combined-{}.csv'.format(ens,m,FF),comment="#", sep="\t").iloc[:, 2][0]
v2e=pd.read_csv('../Results/{}/{}/Fit/Excited-combined-{}.csv'.format(ens,m,FF),comment="#", sep="\t").iloc[:, 2][1]
v3e=pd.read_csv('../Results/{}/{}/Fit/Excited-combined-{}.csv'.format(ens,m,FF),comment="#", sep="\t").iloc[:, 2][2]
v4e=pd.read_csv('../Results/{}/{}/Fit/Excited-combined-{}.csv'.format(ens,m,FF),comment="#", sep="\t").iloc[:, 2][3]
v5e=pd.read_csv('../Results/{}/{}/Fit/Excited-combined-{}.csv'.format(ens,m,FF),comment="#", sep="\t").iloc[:, 2][4]


error=[bs2ptem0,ds2pt0em0,ds2pt1em0,ds2pt2em0,ds2pt3em0,ds2pt4em0,ds2pt5em0,v1e,v2e,v3e,v4e,v5e]

v1p = None
with open('../Results/{}/{}/Fit/Excited-combined-{}.csv'.format(ens,m,FF)) as f:
    for line in f:
        if line.startswith("# pvalue:"):
            # Split at ":" and take the number after it
            v1p = float(line.split(":")[1].strip())
            break




#v1p=pd.read_csv('../Results/{}/{}/Fit/V/pval-V-nsq1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
#v2p=pd.read_csv('../Results/{}/{}/Fit/V/pval-V-nsq2.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
#v3p=pd.read_csv('../Results/{}/{}/Fit/V/pval-V-nsq3.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
#v4p=pd.read_csv('../Results/{}/{}/Fit/V/pval-V-nsq4.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
#v5p=pd.read_csv('../Results/{}/{}/Fit/V/pval-V-nsq5.csv'.format(ens,m), sep='\s')['pval'].iloc[0]


pval=[v1p,v1p,v1p,v1p,v1p,v1p,v1p,v1p,v1p,v1p,v1p,v1p]
df3 = pd.DataFrame(columns=['Value','Error','pval'])
df3['Value']=data
df3['Error']=error
df3['pval']=pval
df3.to_csv('../Results/Crosschecks/AB/Crosscheck-excited-{}-{}-{}-3pt.csv'.format(ens,m,FF), sep='\t')
