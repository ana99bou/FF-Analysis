#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mb = 1.9257122802734448
md_vals = {
    0: 0.73483032,
    1: 0.7458868408203149,
    2: 0.7567413330078149,
    4: 0.77745605,
    5: 0.787656860351565
}
pre = {k: -1/(mb + v) for k, v in md_vals.items()}

# Messdaten
nsq = {
    0: pd.read_csv('Ratios/A1/A1-nsq0.txt', sep=' ', header=None),
    1: pd.read_csv('Ratios/A1/A1-nsq1.txt', sep=' ', header=None),
    2: pd.read_csv('Ratios/A1/A1-nsq2.txt', sep=' ', header=None),
    4: pd.read_csv('Ratios/A1/A1-nsq4.txt', sep=' ', header=None),
    5: pd.read_csv('Ratios/A1/A1-nsq5.txt', sep=' ', header=None)
}
nsq_disp = {
    0: pd.read_csv('Ratios/A1/A1-nsq0-Disp.txt', sep=' ', header=None),
    1: pd.read_csv('Ratios/A1/A1-nsq1-Disp.txt', sep=' ', header=None),
    2: pd.read_csv('Ratios/A1/A1-nsq2-Disp.txt', sep=' ', header=None),
    4: pd.read_csv('Ratios/A1/A1-nsq4-Disp.txt', sep=' ', header=None),
    5: pd.read_csv('Ratios/A1/A1-nsq5-Disp.txt', sep=' ', header=None)
}

# Fits
nsq_fit = {i: pd.read_csv(f'Fits/A1/A1-Av-nsq{i}-Fit.csv', sep='\s') for i in nsq}
nsq_disp_fit = {i: pd.read_csv(f'Fits/A1/A1-Av-nsq{i}-Fit-Disp.csv', sep='\s') for i in nsq_disp}

# Farben
base_colors = {
    0: 'g',
    1: 'b',
    2: 'orange',
    4: 'red',
    5: 'magenta'
}
disp_colors = {
    0: 'limegreen',
    1: 'turquoise',
    2: 'gold',
    4: 'salmon',
    5: 'violet'
}

plt.figure(figsize=(6, 4))
plt.xlabel('Time', fontsize=15)
plt.ylabel(r'$\widetilde{A}_1$', fontsize=15)

# Originaldaten + Fits
for i in nsq:
    plt.errorbar(range(30), nsq[i][0][:30], yerr=nsq[i][1][:30],
                 ls='none', fmt='x', label=fr'$n^2={i}$', color=base_colors[i])
    
    eff = nsq_fit[i]['EffectiveMass']
    sigma = nsq_fit[i]['Error']
    reg_low = int(nsq_fit[i]['RegLow'])
    reg_up = int(nsq_fit[i]['RegUp'])

    plt.plot([-1, 32], [eff, eff], color=base_colors[i], linewidth=0.5)
    plt.fill_between(range(47)[reg_low:reg_up+1], eff + sigma, eff - sigma,
                     color=base_colors[i], alpha=0.2)

# Disp-Daten + Disp-Fits
for i in nsq_disp:
    plt.errorbar(range(30), nsq_disp[i][0][:30], yerr=nsq_disp[i][1][:30],
                 ls='none', fmt='x', color=disp_colors[i], label=fr'$n^2={i}$ Disp')
    
    eff = nsq_disp_fit[i]['EffectiveMass']
    sigma = nsq_disp_fit[i]['Error']
    reg_low = int(nsq_disp_fit[i]['RegLow'])
    reg_up = int(nsq_disp_fit[i]['RegUp'])

    plt.plot([-1, 32], [eff, eff], color=disp_colors[i], linewidth=0.5)
    plt.fill_between(range(47)[reg_low:reg_up+1], eff + sigma, eff - sigma,
                     color=disp_colors[i], alpha=0.2)

plt.annotate(r'$\bf{preliminary}$', xy=(0.17, 0.03), xycoords='axes fraction',
             fontsize=15, color='grey', alpha=.7)

plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=10, ncol=2, markerscale=0.8)
plt.savefig('Niceplot-A1-Disp.pdf', transparent=True, dpi=300, bbox_inches='tight')
'''




import numpy as np
import pandas as pd
#from bokeh.plotting import figure, show, output_file
import matplotlib.pyplot as plt

mb=1.9257122802734448
md0=0.73483032
md1=0.7458868408203149
md2=0.7567413330078149
md4=0.77745605
md5=0.787656860351565
pre1=-1/(mb+md1)
pre2=-1/(mb+md2)
pre0=-1/(mb+md0)
pre4=-1/(mb+md4)
pre5=-1/(mb+md5)

nsq0=pd.read_csv('Ratios/A1/A1-nsq0.txt',sep=' ',header=None)
nsq1=pd.read_csv('Ratios/A1/A1-nsq1.txt', sep=' ', header=None)
nsq2=pd.read_csv('Ratios/A1/A1-nsq2.txt', sep=' ', header=None)
#nsq3=pd.read_csv('Ratios/A1/A1-nsq3.txt', sep=' ', header=None)
nsq4=pd.read_csv('Ratios/A1/A1-nsq4.txt', sep=' ', header=None)
nsq5=pd.read_csv('Ratios/A1/A1-nsq5.txt', sep=' ', header=None)

nsq0plt=pd.read_csv('./Fits/A1/A1-Av-nsq0-Fit.csv',sep='\s')
nsq1plt=pd.read_csv('./Fits/A1/A1-Av-nsq1-Fit.csv',sep='\s')
nsq2plt=pd.read_csv('./Fits/A1/A1-Av-nsq2-Fit.csv',sep='\s')
#nsq3plt=pd.read_csv('./Fits/A1/A1-Av-nsq3-Fit.csv',sep='\s')
nsq4plt=pd.read_csv('./Fits/A1/A1-Av-nsq4-Fit.csv',sep='\s')
nsq5plt=pd.read_csv('./Fits/A1/A1-Av-nsq5-Fit.csv',sep='\s')

x0, y0 = [-1, 32], [nsq0plt['EffectiveMass'], nsq0plt['EffectiveMass']]
x1, y1 = [-1, 32], [nsq1plt['EffectiveMass'], nsq1plt['EffectiveMass']]
x2, y2 = [-1, 32], [nsq2plt['EffectiveMass'], nsq2plt['EffectiveMass']]
#x3, y3 = [-1, 32], [nsq3plt['EffectiveMass'], nsq3plt['EffectiveMass']]
x4, y4 = [-1, 32], [nsq4plt['EffectiveMass'], nsq4plt['EffectiveMass']]
x5, y5 = [-1, 32], [nsq5plt['EffectiveMass'], nsq5plt['EffectiveMass']]

reg_low0=nsq0plt['RegLow']
reg_up0=nsq0plt['RegUp']
sigma0=nsq0plt['Error']
reg_low1=nsq1plt['RegLow']
reg_up1=nsq1plt['RegUp']
sigma1=nsq1plt['Error']
reg_low2=nsq2plt['RegLow']
reg_up2=nsq2plt['RegUp']
sigma2=nsq2plt['Error']
#reg_low3=nsq3plt['RegLow']
#reg_up3=nsq3plt['RegUp']
#sigma3=nsq3plt['Error']
reg_low4=nsq4plt['RegLow']
reg_up4=nsq4plt['RegUp']
sigma4=nsq4plt['Error']
reg_low5=nsq5plt['RegLow']
reg_up5=nsq5plt['RegUp']
sigma5=nsq5plt['Error']

figure_size = (6, 4)
plt.xlabel('Time',fontsize=16)
plt.ylabel(r'$\widetilde{A}_1$',fontsize=16)
#plt.plot(range(96),nsq1[0])
#plt.plot(range(30),avn0y[0:30])
#plt.plot(range(30),avn0z[0:30])
#plt.plot(range(30),avn0[0:30])
#plt.plot(range(96),avn00)

plt.errorbar(list(range(30)), nsq0[0][0:30], yerr=nsq0[1][0:30],ls='none',fmt='x',label='$n^2=0$',color='g')
plt.errorbar(list(range(30)), nsq1[0][0:30], yerr=nsq1[1][0:30],ls='none',fmt='x',label='$n^2=1$',color='b')
plt.errorbar(list(range(30)), nsq2[0][0:30], yerr=nsq2[1][0:30],ls='none',fmt='x',label='$n^2=2$',color='orange')
#plt.errorbar(list(range(30)), nsq3[0][0:30], yerr=nsq3[1][0:30],ls='none',fmt='x',label='$n^2=3$',color='brown')
plt.errorbar(list(range(30)), nsq4[0][0:30], yerr=nsq4[1][0:30],ls='none',fmt='x',label='$n^2=4$',color='red')
plt.errorbar(list(range(30)), nsq5[0][0:30], yerr=nsq5[1][0:30],ls='none',fmt='x',label='$n^2=5$',color='magenta')


plt.plot(x0,y0,color='g')
plt.fill_between(list(range(47))[int(reg_low0):int(reg_up0+1)], nsq0plt['EffectiveMass']+sigma0, nsq0plt['EffectiveMass']-sigma0, color='g',alpha=0.2)

plt.plot(x1,y1, color='b',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low1):int(reg_up1+1)], nsq1plt['EffectiveMass']+sigma1, nsq1plt['EffectiveMass']-sigma1, color='b',alpha=0.2)
plt.plot(x2,y2,color='orange',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low2):int(reg_up2+1)], nsq2plt['EffectiveMass']+sigma2, nsq2plt['EffectiveMass']-sigma2, color='orange',alpha=0.2)
#plt.plot(x3,y3, color='brown',linewidth=0.5)
#plt.fill_between(list(range(47))[int(reg_low1):int(reg_up1+1)], nsq3plt['EffectiveMass']+sigma3, nsq3plt['EffectiveMass']-sigma3, color='brown',alpha=0.2)
plt.plot(x4,y4, color='red',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low4):int(reg_up4+1)], nsq4plt['EffectiveMass']+sigma4, nsq4plt['EffectiveMass']-sigma4, color='red',alpha=0.2)
plt.plot(x5,y5, color='magenta',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low5):int(reg_up5+1)], nsq5plt['EffectiveMass']+sigma5, nsq5plt['EffectiveMass']-sigma5, color='magenta',alpha=0.2)

plt.annotate(r'$\bf{preliminary}$',xy=(0.17,0.03),xycoords='axes fraction',fontsize=15,color='grey',alpha=.7)
#plt.axis((0,30,0,4))
plt.tick_params(axis='both', which='major', labelsize=14)  # For ma
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#plt.yscale('log')
plt.legend(fontsize=14)
plt.savefig('Niceplot-A1.pdf',transparent=True,dpi=300,bbox_inches='tight')
