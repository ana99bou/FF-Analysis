import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ens = 'M2'  # or 'M1', 'M2', etc.
m='0.280'

bs2ptm0=pd.read_csv('../Data/{}/2pt/Excited-comb-BsResult.csv'.format(ens), sep='\s')['Mass0'].iloc[0]
ds2pt0m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-0.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]
ds2pt1m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-1.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]
ds2pt2m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-2.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]
ds2pt3m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-3.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]
ds2pt4m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-4.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]
ds2pt5m0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-5.csv'.format(ens,m), sep='\s')['Mass0'].iloc[0]

bs2ptm1=pd.read_csv('../Data/{}/2pt/Excited-comb-BsResult.csv'.format(ens), sep='\s')['Mass1'].iloc[0]
ds2pt0m1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-0.csv'.format(ens,m), sep='\s')['Mass1'].iloc[0]
ds2pt1m1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-1.csv'.format(ens,m), sep='\s')['Mass1'].iloc[0]
ds2pt2m1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-2.csv'.format(ens,m), sep='\s')['Mass1'].iloc[0]
ds2pt3m1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-3.csv'.format(ens,m), sep='\s')['Mass1'].iloc[0]
ds2pt4m1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-4.csv'.format(ens,m), sep='\s')['Mass1'].iloc[0]
ds2pt5m1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-5.csv'.format(ens,m), sep='\s')['Mass1'].iloc[0]

bs2ptdm=pd.read_csv('../Data/{}/2pt/Excited-comb-BsResult.csv'.format(ens), sep='\s')['DeltaM'].iloc[0]
ds2pt0dm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-0.csv'.format(ens,m), sep='\s')['DeltaM'].iloc[0]
ds2pt1dm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-1.csv'.format(ens,m), sep='\s')['DeltaM'].iloc[0]
ds2pt2dm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-2.csv'.format(ens,m), sep='\s')['DeltaM'].iloc[0]
ds2pt3dm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-3.csv'.format(ens,m), sep='\s')['DeltaM'].iloc[0]
ds2pt4dm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-4.csv'.format(ens,m), sep='\s')['DeltaM'].iloc[0]
ds2pt5dm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-5.csv'.format(ens,m), sep='\s')['DeltaM'].iloc[0]


data=[bs2ptm0,ds2pt0m0,ds2pt1m0,ds2pt2m0,ds2pt3m0,ds2pt4m0,ds2pt5m0,bs2ptm1,ds2pt0m1,ds2pt1m1,ds2pt2m1,ds2pt3m1,ds2pt4m1,ds2pt5m1,bs2ptdm,ds2pt0dm,ds2pt1dm,ds2pt2dm,ds2pt3dm,ds2pt4dm,ds2pt5dm]


bs2ptem0=pd.read_csv('../Data/{}/2pt/Excited-comb-BsResult.csv'.format(ens), sep='\s')['DeltaM0'].iloc[0]
ds2pt0em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-0.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]
ds2pt1em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-1.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]
ds2pt2em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-2.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]
ds2pt3em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-3.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]
ds2pt4em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-4.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]
ds2pt5em0=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-5.csv'.format(ens,m), sep='\s')['DeltaM0'].iloc[0]

bs2ptem1=pd.read_csv('../Data/{}/2pt/Excited-comb-BsResult.csv'.format(ens), sep='\s')['DeltaM1'].iloc[0]
ds2pt0em1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-0.csv'.format(ens,m), sep='\s')['DeltaM1'].iloc[0]
ds2pt1em1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-1.csv'.format(ens,m), sep='\s')['DeltaM1'].iloc[0]
ds2pt2em1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-2.csv'.format(ens,m), sep='\s')['DeltaM1'].iloc[0]
ds2pt3em1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-3.csv'.format(ens,m), sep='\s')['DeltaM1'].iloc[0]
ds2pt4em1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-4.csv'.format(ens,m), sep='\s')['DeltaM1'].iloc[0]
ds2pt5em1=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-5.csv'.format(ens,m), sep='\s')['DeltaM1'].iloc[0]

bs2ptedm=pd.read_csv('../Data/{}/2pt/Excited-comb-BsResult.csv'.format(ens), sep='\s')['DeltaDm'].iloc[0]
ds2pt0edm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-0.csv'.format(ens,m), sep='\s')['DeltaDm'].iloc[0]
ds2pt1edm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-1.csv'.format(ens,m), sep='\s')['DeltaDm'].iloc[0]
ds2pt2edm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-2.csv'.format(ens,m), sep='\s')['DeltaDm'].iloc[0]
ds2pt3edm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-3.csv'.format(ens,m), sep='\s')['DeltaDm'].iloc[0]
ds2pt4edm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-4.csv'.format(ens,m), sep='\s')['DeltaDm'].iloc[0]
ds2pt5edm=pd.read_csv('../Data/{}/2pt/Excited-comb-Ds{}Result-5.csv'.format(ens,m), sep='\s')['DeltaDm'].iloc[0]


error=[bs2ptem0,ds2pt0em0,ds2pt1em0,ds2pt2em0,ds2pt3em0,ds2pt4em0,ds2pt5em0, bs2ptem1,ds2pt0em1,ds2pt1em1,ds2pt2em1,ds2pt3em1,ds2pt4em1,ds2pt5em1,bs2ptedm,ds2pt0edm,ds2pt1edm,ds2pt2edm,ds2pt3edm,ds2pt4edm,ds2pt5edm]


bs2ptp=pd.read_csv('../Data/{}/2pt/Excited-comb-pval-Bs.csv'.format(ens), sep='\s')['p-val'].iloc[0]
ds2pt0p=pd.read_csv('../Data/{}/2pt/Excited-comb-pval-Ds{}-0.csv'.format(ens,m), sep='\s')['p-val'].iloc[0]
ds2pt1p=pd.read_csv('../Data/{}/2pt/Excited-comb-pval-Ds{}-1.csv'.format(ens,m), sep='\s')['p-val'].iloc[0]
ds2pt2p=pd.read_csv('../Data/{}/2pt/Excited-comb-pval-Ds{}-2.csv'.format(ens,m), sep='\s')['p-val'].iloc[0]
ds2pt3p=pd.read_csv('../Data/{}/2pt/Excited-comb-pval-Ds{}-3.csv'.format(ens,m), sep='\s')['p-val'].iloc[0]
ds2pt4p=pd.read_csv('../Data/{}/2pt/Excited-comb-pval-Ds{}-4.csv'.format(ens,m), sep='\s')['p-val'].iloc[0]
ds2pt5p=pd.read_csv('../Data/{}/2pt/Excited-comb-pval-Ds{}-5.csv'.format(ens,m), sep='\s')['p-val'].iloc[0]


pval=[bs2ptp,ds2pt0p,ds2pt1p,ds2pt2p,ds2pt3p,ds2pt4p,ds2pt5p,bs2ptp,ds2pt0p,ds2pt1p,ds2pt2p,ds2pt3p,ds2pt4p,ds2pt5p,bs2ptp,ds2pt0p,ds2pt1p,ds2pt2p,ds2pt3p,ds2pt4p,ds2pt5p]
df3 = pd.DataFrame(columns=['Value','Error','pval'])
df3['Value']=data
df3['Error']=error
df3['pval']=pval
df3.to_csv('../Results/Crosschecks/AB/Crosscheck-excited-{}-{}.csv'.format(ens,m), sep='\t')
