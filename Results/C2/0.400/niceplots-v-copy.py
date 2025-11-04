#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:15:26 2024

@author: anastasiaboushmelev
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameter
mb = 1.9257122802734448
md_vals = {
    1: 0.7458868408203149,
    2: 0.7567413320078149,
    3: 0.767,  # dummy
    4: 0.77745605,
    5: 0.787656860351565
}
pre = {k: -(mb + v) / (2 * mb) for k, v in md_vals.items()}

# Farben
base_colors = {
    1: 'b',
    2: 'orange',
    3: 'brown',
    4: 'red',
    5: 'magenta'
}

colors = plt.cm.tab10.colors
print(colors)
print(base_colors)
disp_colors = {
    1: 'turquoise',
    2: 'gold',
    3: 'peru',
    4: 'salmon',
    5: 'violet'
}

# Messdaten
nsq = {i: pd.read_csv(f'Ratios/V/V-nsq{i}.txt', sep=' ', header=None) for i in base_colors}
nsq_disp = {i: pd.read_csv(f'Ratios/V/V-nsq{i}-Disp.txt', sep=' ', header=None) for i in disp_colors}

# Fits
nsq_fit = {i: pd.read_csv(f'Fits/V/V-Av-nsq{i}-Fit.csv', sep='\s') for i in base_colors}
nsq_disp_fit = {i: pd.read_csv(f'Fits/V/V-Av-nsq{i}-Fit-Disp.csv', sep='\s') for i in disp_colors}

plt.figure(figsize=(6, 4))
plt.xlabel('Time', fontsize=15)
plt.ylabel(r'$\widetilde{V}$', fontsize=15)

# Normale Daten
for i in nsq:
    plt.errorbar(range(1, 20), nsq[i][0][1:20], yerr=nsq[i][1][1:20],
                 ls='none', fmt='x', label=fr'$n^2={i}$', color=base_colors[i])
    
    fit = nsq_fit[i]
    eff, sigma = fit['EffectiveMass'], fit['Error']
    reg_low, reg_up = int(fit['RegLow']), int(fit['RegUp'])

    plt.plot([-1, 22], [eff, eff], color=base_colors[i], linewidth=0.5)
    plt.fill_between(range(47)[reg_low:reg_up+1], eff + sigma, eff - sigma,
                     color=base_colors[i], alpha=0.2)

# Disp-Daten
for i in nsq_disp:
    plt.errorbar(range(1, 20), nsq_disp[i][0][1:20], yerr=nsq_disp[i][1][1:20],
                 ls='none', fmt='x', label=fr'$n^2={i}$ Disp', color=disp_colors[i])
    
    fit = nsq_disp_fit[i]
    eff, sigma = fit['EffectiveMass'], fit['Error']
    reg_low, reg_up = int(fit['RegLow']), int(fit['RegUp'])

    plt.plot([-1, 22], [eff, eff], color=disp_colors[i], linewidth=0.5)
    plt.fill_between(range(47)[reg_low:reg_up+1], eff + sigma, eff - sigma,
                     color=disp_colors[i], alpha=0.2)

plt.annotate(r'$\bf{preliminary}$', xy=(0.17, 0.03), xycoords='axes fraction',
             fontsize=15, color='grey', alpha=.7)

plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(np.arange(0, int(20) + 1, 5))  # force integer x-ticks
    
plt.legend(fontsize=10, ncol=2, markerscale=0.8)
plt.savefig('Niceplot-V-Disp.pdf', transparent=True, dpi=200, bbox_inches='tight')



'''
import numpy as np
import pandas as pd
#from bokeh.plotting import figure, show, output_file
import matplotlib.pyplot as plt

mb=1.9257122802734448
md0=0.73482022
md1=0.7458868408203149
md2=0.7567413320078149
md4=0.77745605
pre1=-(mb+md0)/(2*mb)
pre2=pre1
pre4=pre1

nsq3=pd.read_csv('Ratios/V/V-nsq3.txt', sep=' ', header=None)
nsq1=pd.read_csv('Ratios/V/V-nsq1.txt', sep=' ', header=None)
nsq2=pd.read_csv('Ratios/V/V-nsq2.txt', sep=' ', header=None)
nsq4=pd.read_csv('Ratios/V/V-nsq4.txt', sep=' ', header=None)
nsq5=pd.read_csv('Ratios/V/V-nsq5.txt', sep=' ', header=None)

#nsq0plt=pd.read_csv('./Fits/hA1-Av-nsq0-Fit.csv',sep='\s')
nsq1plt=pd.read_csv('Fits/V/V-Av-nsq1-Fit.csv', sep='\s')
nsq2plt=pd.read_csv('Fits/V/V-Av-nsq2-Fit.csv', sep='\s')
nsq3plt=pd.read_csv('Fits/V/V-Av-nsq3-Fit.csv', sep='\s')
nsq4plt=pd.read_csv('Fits/V/V-Av-nsq4-Fit.csv', sep='\s')
nsq5plt=pd.read_csv('Fits/V/V-Av-nsq5-Fit.csv', sep='\s')

#x0, y0 = [-1, 22], [nsq0plt['EffectiveMass'], nsq0plt['EffectiveMass']]
x1, y1 = [-1, 22], [nsq1plt['EffectiveMass'], nsq1plt['EffectiveMass']]
x2, y2 = [-1, 22], [nsq2plt['EffectiveMass'], nsq2plt['EffectiveMass']]
x3, y3 = [-1, 22], [nsq3plt['EffectiveMass'], nsq3plt['EffectiveMass']]
x4, y4 = [-1, 22], [nsq4plt['EffectiveMass'],nsq4plt['EffectiveMass']]
x5, y5 = [-1, 22], [nsq5plt['EffectiveMass'],nsq5plt['EffectiveMass']]

reg_low1=nsq1plt['RegLow']
reg_up1=nsq1plt['RegUp']
sigma1=nsq1plt['Error']
reg_low2=nsq2plt['RegLow']
reg_up2=nsq2plt['RegUp']
sigma2=nsq2plt['Error']
reg_low3=nsq3plt['RegLow']
reg_up3=nsq3plt['RegUp']
sigma3=nsq3plt['Error']
reg_low4=nsq4plt['RegLow']
reg_up4=nsq4plt['RegUp']
sigma4=nsq4plt['Error']
reg_low5=nsq5plt['RegLow']
reg_up5=nsq5plt['RegUp']
sigma5=nsq5plt['Error']
#reg_low0=nsq0plt['RegLow']
#reg_up0=nsq0plt['RegUp']
#sigma0=nsq0plt['Error']

figure_size = (6, 4)
plt.xlabel('Time',fontsize=15)
plt.ylabel(r'$\widetilde{V}$',fontsize=15)
#plt.plot(range(96),nsq1[0])
#plt.plot(range(20),avn0y[0:20])
#plt.plot(range(20),avn0z[0:20])
#plt.plot(range(20),avn0[0:20])
#plt.plot(range(96),avn00)
#plt.errorbar(list(range(20)), pre0*nsq0[0][0:20], yerr=pre0*nsq0[1][0:20],ls='none',fmt='x',label='$n^2=0$',color='g')

plt.errorbar(list(range(29)), nsq1[0][1:20], yerr=nsq1[1][1:20],ls='none',fmt='x',label='$n^2=1$',color='b')
plt.errorbar(list(range(29)), nsq2[0][1:20], yerr=nsq2[1][1:20],ls='none',fmt='x',label='$n^2=2$',color='orange')
plt.errorbar(list(range(29)), nsq3[0][1:20], yerr=nsq3[1][1:20],ls='none',fmt='x',label='$n^2=3$',color='brown')
plt.errorbar(list(range(29)), nsq4[0][1:20], yerr=nsq4[1][1:20],ls='none',fmt='x',label='$n^2=4$',color='red')
plt.errorbar(list(range(29)), nsq5[0][1:20], yerr=nsq5[1][1:20],ls='none',fmt='x',label='$n^2=5$',color='magenta')

#plt.plot(x0,y0,color='g')
#plt.fill_between(list(range(47))[int(reg_low0):int(reg_up0+1)], nsq0plt['EffectiveMass']+sigma0, nsq0plt['EffectiveMass']-sigma0, color='g',alpha=0.2)

plt.plot(x1,y1, color='b',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low1):int(reg_up1+1)], nsq1plt['EffectiveMass']+sigma1, nsq1plt['EffectiveMass']-sigma1, color='b',alpha=0.2)
plt.plot(x2,y2,color='orange',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low2):int(reg_up2+1)], nsq2plt['EffectiveMass']+sigma2, nsq2plt['EffectiveMass']-sigma2, color='orange',alpha=0.2)
plt.annotate(r'$\bf{preliminary}$',xy=(0.17,0.03),xycoords='axes fraction',fontsize=15,color='grey',alpha=.7)
plt.fill_between(list(range(47))[int(reg_low3):int(reg_up3+1)], nsq3plt['EffectiveMass']+sigma3, nsq3plt['EffectiveMass']-sigma3, color='brown',alpha=0.2)
plt.plot(x3,y3,color='brown',linewidth=0.5)
plt.plot(x4,y4, color='red',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low4):int(reg_up4+1)], (nsq4plt['EffectiveMass']+sigma4), (nsq4plt['EffectiveMass']-sigma4), color='red',alpha=0.2)
plt.plot(x5,y5, color='magenta',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low5):int(reg_up5+1)], (nsq5plt['EffectiveMass']+sigma5), (nsq5plt['EffectiveMass']-sigma5), color='magenta',alpha=0.2)

plt.tick_params(axis='both', which='major', labelsize=14) 
#plt.axis((0,20,0.05,0.11))

#plt.yscale('log')
plt.legend()
plt.savefig('Niceplot-V.pdf',transparent=True,dpi=200,bbox_inches='tight')
'''