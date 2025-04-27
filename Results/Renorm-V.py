import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

crho_V0 = 1.0494#(119)
crho_Vi = 1.0138#(19)
mrho_V0 = 1.0366#(62)
mrho_Vi = 1.0106#(22)
frho_V0 = 1.0318#(43) 
frho_Vi = 1.0098#(20)

C1Zall=0.7172
C1Zacc=0.85134
C1ZVbb=9.099

M2Zall=0.7452
M2Zacc=0.802280
M2ZVbb=4.7511

F1SZall=0.7624
F1SZacc=0.78812 ##########Not yet correct
F1SZVbb=3.622



mbf1s=pd.read_csv('../Data/F1S/2pt/BsResult.csv', sep='\s')['EffectiveMass'].iloc[0]
md0f1s=pd.read_csv('../Data/F1S/2pt/Ds0.248Result-0.csv', sep='\s')['EffectiveMass'].iloc[0]
md1f1s=pd.read_csv('../Data/F1S/2pt/Ds0.248Result-1.csv', sep='\s')['EffectiveMass'].iloc[0]
md2f1s=pd.read_csv('../Data/F1S/2pt/Ds0.248Result-2.csv', sep='\s')['EffectiveMass'].iloc[0]
md3f1s=pd.read_csv('../Data/F1S/2pt/Ds0.248Result-3.csv', sep='\s')['EffectiveMass'].iloc[0]
md4f1s=pd.read_csv('../Data/F1S/2pt/Ds0.248Result-4.csv', sep='\s')['EffectiveMass'].iloc[0]
md5f1s=pd.read_csv('../Data/F1S/2pt/Ds0.248Result-5.csv', sep='\s')['EffectiveMass'].iloc[0]


mbc1=pd.read_csv('../Data/C1/2pt/BsResult.csv', sep='\s')['EffectiveMass'].iloc[0]
md0c1=pd.read_csv('../Data/C1/2pt/Ds0.400Result-0.csv', sep='\s')['EffectiveMass'].iloc[0]
md1c1=pd.read_csv('../Data/C1/2pt/Ds0.400Result-1.csv', sep='\s')['EffectiveMass'].iloc[0]
md2c1=pd.read_csv('../Data/C1/2pt/Ds0.400Result-2.csv', sep='\s')['EffectiveMass'].iloc[0]
md3c1=pd.read_csv('../Data/C1/2pt/Ds0.400Result-3.csv', sep='\s')['EffectiveMass'].iloc[0]
md4c1=pd.read_csv('../Data/C1/2pt/Ds0.400Result-4.csv', sep='\s')['EffectiveMass'].iloc[0]
md5c1=pd.read_csv('../Data/C1/2pt/Ds0.400Result-5.csv', sep='\s')['EffectiveMass'].iloc[0]


mbm2=pd.read_csv('../Data/M2/2pt/BsResult.csv', sep='\s')['EffectiveMass'].iloc[0]
md0m2=pd.read_csv('../Data/M2/2pt/Ds0.340Result-0.csv', sep='\s')['EffectiveMass'].iloc[0]
md1m2=pd.read_csv('../Data/M2/2pt/Ds0.340Result-1.csv', sep='\s')['EffectiveMass'].iloc[0]
md2m2=pd.read_csv('../Data/M2/2pt/Ds0.340Result-2.csv', sep='\s')['EffectiveMass'].iloc[0]
md3m2=pd.read_csv('../Data/M2/2pt/Ds0.340Result-3.csv', sep='\s')['EffectiveMass'].iloc[0]
md4m2=pd.read_csv('../Data/M2/2pt/Ds0.340Result-4.csv', sep='\s')['EffectiveMass'].iloc[0]
md5m2=pd.read_csv('../Data/M2/2pt/Ds0.340Result-5.csv', sep='\s')['EffectiveMass'].iloc[0]


f1sr=[2.785**2*(mbf1s**2+md0f1s**2-2*mbf1s*md0f1s),2.785**2*(mbf1s**2+md0f1s**2-2*mbf1s*md1f1s),2.785**2*(mbf1s**2+md0f1s**2-2*mbf1s*md2f1s),2.785**2*(mbf1s**2+md0f1s**2-2*mbf1s*md3f1s),2.785**2*(mbf1s**2+md0f1s**2-2*mbf1s*md4f1s),2.785**2*(mbf1s**2+md0f1s**2-2*mbf1s*md5f1s)]
c1r=[1.7848**2*(mbc1**2+md0c1**2-2*mbc1*md0c1),1.7848**2*(mbc1**2+md0c1**2-2*mbc1*md1c1),1.7848**2*(mbc1**2+md0c1**2-2*mbc1*md2c1),1.7848**2*(mbc1**2+md0c1**2-2*mbc1*md3c1),1.7848**2*(mbc1**2+md0c1**2-2*mbc1*md4c1),1.7848**2*(mbc1**2+md0c1**2-2*mbc1*md5c1)]
m2r=[2.3833**2*(mbm2**2+md0m2**2-2*mbm2*md0m2),2.3833**2*(mbm2**2+md0m2**2-2*mbm2*md1m2),2.3833**2*(mbm2**2+md0m2**2-2*mbm2*md2m2),2.3833**2*(mbm2**2+md0m2**2-2*mbm2*md3m2),2.3833**2*(mbm2**2+md0m2**2-2*mbm2*md4m2),2.3833**2*(mbm2**2+md0m2**2-2*mbm2*md5m2)]

print(md0c1*1.7848,md0m2*2.3833,md0f1s*2.785)

print((mbc1**2+md0c1**2-2*mbc1*md0c1))
print((mbm2**2+md0m2**2-2*mbm2*md0m2))
print((mbf1s**2+md0f1s**2-2*mbf1s*md0f1s))

print(1.7848**2*(mbc1**2+md0c1**2-2*mbc1*md0c1))
print(2.3833**2*(mbm2**2+md0m2**2-2*mbm2*md0m2))
print(2.785**2*(mbf1s**2+md0f1s**2-2*mbf1s*md0f1s))


c1x_coords = [c1r[1], c1r[2], c1r[3],c1r[4],c1r[5]]
f1sx_coords = [f1sr[1], f1sr[2], f1sr[3],f1sr[4],f1sr[5]]
m2x_coords =[m2r[1], m2r[2], m2r[3],m2r[4],m2r[5]]

print(c1x_coords)
print(m2x_coords)
print(f1sx_coords)


c1nsq1plt=pd.read_csv('./C1/Fits/V/V-Av-nsq1-Fit.csv', sep='\s')
c1nsq2plt=pd.read_csv('./C1/Fits/V/V-Av-nsq2-Fit.csv', sep='\s')
c1nsq3plt=pd.read_csv('./C1/Fits/V/V-Av-nsq3-Fit.csv', sep='\s')
c1nsq4plt=pd.read_csv('./C1/Fits/V/V-Av-nsq4-Fit.csv', sep='\s')
c1nsq5plt=pd.read_csv('./C1/Fits/V/V-Av-nsq5-Fit.csv', sep='\s')

m2nsq1plt=pd.read_csv('./M2/Fits/V/V-Av-nsq1-Fit.csv', sep='\s')
m2nsq2plt=pd.read_csv('./M2/Fits/V/V-Av-nsq2-Fit.csv', sep='\s')
m2nsq3plt=pd.read_csv('./M2/Fits/V/V-Av-nsq3-Fit.csv', sep='\s')
m2nsq4plt=pd.read_csv('./M2/Fits/V/V-Av-nsq4-Fit.csv', sep='\s')
m2nsq5plt=pd.read_csv('./M2/Fits/V/V-Av-nsq5-Fit.csv', sep='\s')

f1snsq1plt=pd.read_csv('./F1S/Fits/V/V-Av-nsq1-Fit.csv', sep='\s')
f1snsq2plt=pd.read_csv('./F1S/Fits/V/V-Av-nsq2-Fit.csv', sep='\s')
f1snsq3plt=pd.read_csv('./F1S/Fits/V/V-Av-nsq3-Fit.csv', sep='\s')
f1snsq4plt=pd.read_csv('./F1S/Fits/V/V-Av-nsq4-Fit.csv', sep='\s')
f1snsq5plt=pd.read_csv('./F1S/Fits/V/V-Av-nsq5-Fit.csv', sep='\s')


v_coords_c1=[c1nsq1plt['EffectiveMass'].iloc[0],c1nsq2plt['EffectiveMass'].iloc[0],c1nsq3plt['EffectiveMass'].iloc[0],c1nsq4plt['EffectiveMass'].iloc[0],c1nsq5plt['EffectiveMass'].iloc[0]]
v_errors_c1=[c1nsq1plt['Error'].iloc[0],c1nsq2plt['Error'].iloc[0],c1nsq3plt['Error'].iloc[0],c1nsq4plt['Error'].iloc[0],c1nsq5plt['Error'].iloc[0]]

v_coords_f1s=[f1snsq1plt['EffectiveMass'].iloc[0],f1snsq2plt['EffectiveMass'].iloc[0],f1snsq3plt['EffectiveMass'].iloc[0],f1snsq4plt['EffectiveMass'].iloc[0],f1snsq5plt['EffectiveMass'].iloc[0]]
v_errors_f1s=[f1snsq1plt['Error'].iloc[0],f1snsq2plt['Error'].iloc[0],f1snsq3plt['Error'].iloc[0],f1snsq4plt['Error'].iloc[0],f1snsq5plt['Error'].iloc[0]]

v_coords_m2=[m2nsq1plt['EffectiveMass'].iloc[0],m2nsq2plt['EffectiveMass'].iloc[0],m2nsq3plt['EffectiveMass'].iloc[0],m2nsq4plt['EffectiveMass'].iloc[0],m2nsq5plt['EffectiveMass'].iloc[0]]
v_errors_m2=[m2nsq1plt['Error'].iloc[0],m2nsq2plt['Error'].iloc[0],m2nsq3plt['Error'].iloc[0],m2nsq4plt['Error'].iloc[0],m2nsq5plt['Error'].iloc[0]]

print(v_coords_c1)

v_coords_c1 = [crho_Vi*np.sqrt(C1Zacc*C1ZVbb)*x for x in v_coords_c1]
v_coords_f1s = [frho_Vi*np.sqrt(F1SZacc*F1SZVbb)*x for x in v_coords_f1s]
v_coords_m2 = [mrho_Vi*np.sqrt(M2Zacc*M2ZVbb)*x for x in v_coords_m2]



plt.errorbar(c1x_coords, v_coords_c1, yerr=v_errors_c1, fmt='o', color='red',ecolor='red', capsize=5, capthick=2, elinewidth=1,label=r'C1, $m_c=a0.400$')
plt.errorbar(f1sx_coords, v_coords_f1s, yerr=v_errors_f1s, fmt='o', color='green',ecolor='green', capsize=5, capthick=2, elinewidth=1,label=r'F1S, $m_c=a0.248$')
plt.errorbar(m2x_coords, v_coords_m2, yerr=v_errors_m2, fmt='o', color='blue',ecolor='blue', capsize=5, capthick=2, elinewidth=1,label=r'M2, $m_c=a0.340$')

#plt.axis((0,12,0.8,1.4))

plt.xlabel(r'$q^2[GeV]^2$',fontsize=15)
plt.ylabel(r'V',fontsize=15)

plt.tick_params(axis='both', which='major', labelsize=14) 
plt.legend()
plt.annotate(r'$\bf{preliminary}$',xy=(0.65,0.03),xycoords='axes fraction',fontsize=15,color='grey',alpha=1)

# Show plot
#plt.grid(True)
plt.savefig('FF-q^2.pdf',transparent=True,bbox_inches='tight')