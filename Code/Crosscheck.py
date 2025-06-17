import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ens = 'M1'  # or 'M1', 'M2', etc.
m='0.280'

bs2pt=pd.read_csv('../Data/{}/2pt/BsResult.csv'.format(ens), sep='\s')['EffectiveMass'].iloc[0]
ds2pt0=pd.read_csv('../Data/{}/2pt/Ds{}Result-0.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
ds2pt1=pd.read_csv('../Data/{}/2pt/Ds{}Result-1.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
ds2pt2=pd.read_csv('../Data/{}/2pt/Ds{}Result-2.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
ds2pt3=pd.read_csv('../Data/{}/2pt/Ds{}Result-3.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
ds2pt4=pd.read_csv('../Data/{}/2pt/Ds{}Result-4.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
ds2pt5=pd.read_csv('../Data/{}/2pt/Ds{}Result-5.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]

v1=pd.read_csv('../Results/{}/{}/Fits/V/V-Av-nsq1-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
v2=pd.read_csv('../Results/{}/{}/Fits/V/V-Av-nsq2-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
v3=pd.read_csv('../Results/{}/{}/Fits/V/V-Av-nsq3-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
v4=pd.read_csv('../Results/{}/{}/Fits/V/V-Av-nsq4-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
v5=pd.read_csv('../Results/{}/{}/Fits/V/V-Av-nsq5-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]

a01=pd.read_csv('../Results/{}/{}/Fits/A0/A0-Av-nsq1-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
a02=pd.read_csv('../Results/{}/{}/Fits/A0/A0-Av-nsq2-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
a03=pd.read_csv('../Results/{}/{}/Fits/A0/A0-Av-nsq3-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
a04=pd.read_csv('../Results/{}/{}/Fits/A0/A0-Av-nsq4-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
a05=pd.read_csv('../Results/{}/{}/Fits/A0/A0-Av-nsq5-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]

a10=pd.read_csv('../Results/{}/{}/Fits/A1/A1-Av-nsq0-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
a11=pd.read_csv('../Results/{}/{}/Fits/A1/A1-Av-nsq1-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
a12=pd.read_csv('../Results/{}/{}/Fits/A1/A1-Av-nsq2-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
a14=pd.read_csv('../Results/{}/{}/Fits/A1/A1-Av-nsq4-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]
a15=pd.read_csv('../Results/{}/{}/Fits/A1/A1-Av-nsq5-Fit.csv'.format(ens,m), sep='\s')['EffectiveMass'].iloc[0]

data=[bs2pt,ds2pt0,ds2pt1,ds2pt2,ds2pt3,ds2pt4,ds2pt5,v1,v2,v3,v4,v5,a01,a02,a03,a04,a05,a10,a11,a12,a14,a15]


bs2pte=pd.read_csv('../Data/{}/2pt/BsResult.csv'.format(ens), sep='\s')['Error'].iloc[0]
ds2pt0e=pd.read_csv('../Data/{}/2pt/Ds{}Result-0.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
ds2pt1e=pd.read_csv('../Data/{}/2pt/Ds{}Result-1.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
ds2pt2e=pd.read_csv('../Data/{}/2pt/Ds{}Result-2.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
ds2pt3e=pd.read_csv('../Data/{}/2pt/Ds{}Result-3.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
ds2pt4e=pd.read_csv('../Data/{}/2pt/Ds{}Result-4.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
ds2pt5e=pd.read_csv('../Data/{}/2pt/Ds{}Result-5.csv'.format(ens,m), sep='\s')['Error'].iloc[0]

v1e=pd.read_csv('../Results/{}/{}/Fits/V/V-Av-nsq1-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
v2e=pd.read_csv('../Results/{}/{}/Fits/V/V-Av-nsq2-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
v3e=pd.read_csv('../Results/{}/{}/Fits/V/V-Av-nsq3-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
v4e=pd.read_csv('../Results/{}/{}/Fits/V/V-Av-nsq4-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
v5e=pd.read_csv('../Results/{}/{}/Fits/V/V-Av-nsq5-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]

a01e=pd.read_csv('../Results/{}/{}/Fits/A0/A0-Av-nsq1-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
a02e=pd.read_csv('../Results/{}/{}/Fits/A0/A0-Av-nsq2-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
a03e=pd.read_csv('../Results/{}/{}/Fits/A0/A0-Av-nsq3-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
a04e=pd.read_csv('../Results/{}/{}/Fits/A0/A0-Av-nsq4-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
a05e=pd.read_csv('../Results/{}/{}/Fits/A0/A0-Av-nsq5-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]

a10e=pd.read_csv('../Results/{}/{}/Fits/A1/A1-Av-nsq0-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
a11e=pd.read_csv('../Results/{}/{}/Fits/A1/A1-Av-nsq1-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
a12e=pd.read_csv('../Results/{}/{}/Fits/A1/A1-Av-nsq2-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
a14e=pd.read_csv('../Results/{}/{}/Fits/A1/A1-Av-nsq4-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]
a15e=pd.read_csv('../Results/{}/{}/Fits/A1/A1-Av-nsq5-Fit.csv'.format(ens,m), sep='\s')['Error'].iloc[0]

error=[bs2pte,ds2pt0e,ds2pt1e,ds2pt2e,ds2pt3e,ds2pt4e,ds2pt5e,v1e,v2e,v3e,v4e,v5e,a01e,a02e,a03e,a04e,a05e,a10e,a11e,a12e,a14e,a15e]


bs2ptp=pd.read_csv('../Data/{}/2pt/pval-Bs.csv'.format(ens), sep='\s')['pval'].iloc[0]
ds2pt0p=pd.read_csv('../Data/{}/2pt/pval-Ds{}-0.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
ds2pt1p=pd.read_csv('../Data/{}/2pt/pval-Ds{}-1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
ds2pt2p=pd.read_csv('../Data/{}/2pt/pval-Ds{}-2.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
ds2pt3p=pd.read_csv('../Data/{}/2pt/pval-Ds{}-3.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
ds2pt4p=pd.read_csv('../Data/{}/2pt/pval-Ds{}-4.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
ds2pt5p=pd.read_csv('../Data/{}/2pt/pval-Ds{}-5.csv'.format(ens,m), sep='\s')['pval'].iloc[0]

v1p=pd.read_csv('../Results/{}/{}/Fits/V/pval-V-nsq1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
v2p=pd.read_csv('../Results/{}/{}/Fits/V/pval-V-nsq2.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
v3p=pd.read_csv('../Results/{}/{}/Fits/V/pval-V-nsq3.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
v4p=pd.read_csv('../Results/{}/{}/Fits/V/pval-V-nsq4.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
v5p=pd.read_csv('../Results/{}/{}/Fits/V/pval-V-nsq5.csv'.format(ens,m), sep='\s')['pval'].iloc[0]

a01p=pd.read_csv('../Results/{}/{}/Fits/A0/pval-A0-nsq1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
a02p=pd.read_csv('../Results/{}/{}/Fits/A0/pval-A0-nsq1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
a03p=pd.read_csv('../Results/{}/{}/Fits/A0/pval-A0-nsq1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
a04p=pd.read_csv('../Results/{}/{}/Fits/A0/pval-A0-nsq1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
a05p=pd.read_csv('../Results/{}/{}/Fits/A0/pval-A0-nsq1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]

a10p=pd.read_csv('../Results/{}/{}/Fits/A1/pval-A1-nsq0.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
a11p=pd.read_csv('../Results/{}/{}/Fits/A1/pval-A1-nsq1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
a12p=pd.read_csv('../Results/{}/{}/Fits/A1/pval-A1-nsq1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
a14p=pd.read_csv('../Results/{}/{}/Fits/A1/pval-A1-nsq1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]
a15p=pd.read_csv('../Results/{}/{}/Fits/A1/pval-A1-nsq1.csv'.format(ens,m), sep='\s')['pval'].iloc[0]


pval=[bs2ptp,ds2pt0p,ds2pt1p,ds2pt2p,ds2pt3p,ds2pt4p,ds2pt5p,v1p,v2p,v3p,v4p,v5p,a01p,a02p,a03p,a04p,a05p,a10p,a11p,a12p,a14p,a15p]
df3 = pd.DataFrame(columns=['Value','Error','pval'])
df3['Value']=data
df3['Error']=error
df3['pval']=pval
df3.to_csv('../Results/Crosscheck-{}-{}.csv'.format(ens,m), sep='\t')
