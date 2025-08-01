#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:15:26 2524

@author: anastasiaboushmelev
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:15:26 2524

@author: anastasiaboushmelev
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:15:26 2524

@author: anastasiaboushmelev
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mb = 1.9257127802734448
md0 = 0.73482527
md1 = 0.7458868408253149
md2 = 0.7567412775078149
md4 = 0.77745605
md5 = 0.787656860351565
pre1 = -1/(mb + md1)
pre2 = -1/(mb + md2)
pre0 = -1/(mb + md0)
pre4 = -1/(mb + md4)
pre5 = -1/(mb + md5)

# Original Daten
nsq1 = pd.read_csv('Ratios/A0/A0-nsq1.txt', sep=' ', header=None)
nsq2 = pd.read_csv('Ratios/A0/A0-nsq2.txt', sep=' ', header=None)
nsq3 = pd.read_csv('Ratios/A0/A0-nsq3.txt', sep=' ', header=None)
nsq4 = pd.read_csv('Ratios/A0/A0-nsq4.txt', sep=' ', header=None)
nsq5 = pd.read_csv('Ratios/A0/A0-nsq5.txt', sep=' ', header=None)

nsq1plt = pd.read_csv('./Fits/A0/A0-Av-nsq1-Fit.csv', sep='\s')
nsq2plt = pd.read_csv('./Fits/A0/A0-Av-nsq2-Fit.csv', sep='\s')
nsq3plt = pd.read_csv('./Fits/A0/A0-Av-nsq3-Fit.csv', sep='\s')
nsq4plt = pd.read_csv('./Fits/A0/A0-Av-nsq4-Fit.csv', sep='\s')
nsq5plt = pd.read_csv('./Fits/A0/A0-Av-nsq5-Fit.csv', sep='\s')

# Dispersion Daten
nsq1_disp = pd.read_csv('Ratios/A0/A0-nsq1-Disp.txt', sep=' ', header=None)
nsq2_disp = pd.read_csv('Ratios/A0/A0-nsq2-Disp.txt', sep=' ', header=None)
nsq3_disp = pd.read_csv('Ratios/A0/A0-nsq3-Disp.txt', sep=' ', header=None)
nsq4_disp = pd.read_csv('Ratios/A0/A0-nsq4-Disp.txt', sep=' ', header=None)
nsq5_disp = pd.read_csv('Ratios/A0/A0-nsq5-Disp.txt', sep=' ', header=None)

nsq1plt_disp = pd.read_csv('./Fits/A0/A0-Av-nsq1-Fit-Disp.csv', sep='\s')
nsq2plt_disp = pd.read_csv('./Fits/A0/A0-Av-nsq2-Fit-Disp.csv', sep='\s')
nsq3plt_disp = pd.read_csv('./Fits/A0/A0-Av-nsq3-Fit-Disp.csv', sep='\s')
nsq4plt_disp = pd.read_csv('./Fits/A0/A0-Av-nsq4-Fit-Disp.csv', sep='\s')
nsq5plt_disp = pd.read_csv('./Fits/A0/A0-Av-nsq5-Fit-Disp.csv', sep='\s')

# Farben definieren
colors = {
    1: 'b',
    2: 'orange',
    3: 'brown',
    4: 'red',
    5: 'magenta'
}
disp_colors = {
    1: 'turquoise',
    2: 'gold',
    3: 'peru',
    4: 'salmon',
    5: 'orchid'
}

# Fitregionen und Fehler
reg_low1 = nsq1plt['RegLow']
reg_up1 = nsq1plt['RegUp']
sigma1 = nsq1plt['Error']
reg_low2 = nsq2plt['RegLow']
reg_up2 = nsq2plt['RegUp']
sigma2 = nsq2plt['Error']
reg_low3 = nsq3plt['RegLow']
reg_up3 = nsq3plt['RegUp']
sigma3 = nsq3plt['Error']
reg_low4 = nsq4plt['RegLow']
reg_up4 = nsq4plt['RegUp']
sigma4 = nsq4plt['Error']
reg_low5 = nsq5plt['RegLow']
reg_up5 = nsq5plt['RegUp']
sigma5 = nsq5plt['Error']

# Plot
plt.figure(figsize=(6, 4))
plt.xlabel('Time', fontsize=15)
plt.ylabel(r'$\widetilde{A}_0$', fontsize=15)

# Original Datenpunkte
plt.errorbar(range(25), nsq1[0][:25], yerr=nsq1[1][:25], fmt='x', ls='none', color=colors[1], label='$n^2=1$')
plt.errorbar(range(25), nsq2[0][:25], yerr=nsq2[1][:25], fmt='x', ls='none', color=colors[2], label='$n^2=2$')
plt.errorbar(range(25), nsq3[0][:25], yerr=nsq3[1][:25], fmt='x', ls='none', color=colors[3], label='$n^2=3$')
plt.errorbar(range(25), nsq4[0][:25], yerr=nsq4[1][:25], fmt='x', ls='none', color=colors[4], label='$n^2=4$')
plt.errorbar(range(25), nsq5[0][:25], yerr=nsq5[1][:25], fmt='x', ls='none', color=colors[5], label='$n^2=5$')

# Dispersion Datenpunkte
plt.errorbar(range(25), nsq1_disp[0][:25], yerr=nsq1_disp[1][:25], fmt='x', ls='none', color=disp_colors[1], label='$n^2=1$ Disp')
plt.errorbar(range(25), nsq2_disp[0][:25], yerr=nsq2_disp[1][:25], fmt='x', ls='none', color=disp_colors[2], label='$n^2=2$ Disp')
plt.errorbar(range(25), nsq3_disp[0][:25], yerr=nsq3_disp[1][:25], fmt='x', ls='none', color=disp_colors[3], label='$n^2=3$ Disp')
plt.errorbar(range(25), nsq4_disp[0][:25], yerr=nsq4_disp[1][:25], fmt='x', ls='none', color=disp_colors[4], label='$n^2=4$ Disp')
plt.errorbar(range(25), nsq5_disp[0][:25], yerr=nsq5_disp[1][:25], fmt='x', ls='none', color=disp_colors[5], label='$n^2=5$ Disp')

# Fits + Bänder (Original)
for i, (c, plt_data, sigma, rlow, rup) in enumerate(zip(
    colors.values(),
    [nsq1plt, nsq2plt, nsq3plt, nsq4plt, nsq5plt],
    [sigma1, sigma2, sigma3, sigma4, sigma5],
    [reg_low1, reg_low2, reg_low3, reg_low4, reg_low5],
    [reg_up1, reg_up2, reg_up3, reg_up4, reg_up5]
)):
    em = plt_data['EffectiveMass']
    plt.plot([-1, 27], [em, em], color=c, linewidth=0.5)
    plt.fill_between(range(47)[int(rlow):int(rup + 1)], em + sigma, em - sigma, color=c, alpha=0.2)

# Fits + Bänder (Disp)
for i, (c, plt_data) in enumerate(zip(disp_colors.values(), [
    nsq1plt_disp, nsq2plt_disp, nsq3plt_disp, nsq4plt_disp, nsq5plt_disp])):
    em = plt_data['EffectiveMass']
    sigma = plt_data['Error']
    rlow = int(plt_data['RegLow'])
    rup = int(plt_data['RegUp'])
    plt.plot([-1, 27], [em, em], color=c, linewidth=0.5)
    plt.fill_between(range(47)[rlow:rup + 1], em + sigma, em - sigma, color=c, alpha=0.2)

# Rest
plt.axis((0,25,0.1,0.3))
plt.annotate(r'$\bf{preliminary}$', xy=(0.17, 0.03), xycoords='axes fraction',
             fontsize=15, color='grey', alpha=.7)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=7)  
plt.savefig('Niceplot-A0-Disp.pdf', transparent=True, dpi=250, bbox_inches='tight')


'''
import numpy as np
import pandas as pd
#from bokeh.plotting import figure, show, output_file
import matplotlib.pyplot as plt

mb=1.9257127802734448
md0=0.73482527
md1=0.7458868408253149
md2=0.7567412775078149
md4=0.77745605
md5=0.787656860351565
pre1=-1/(mb+md1)
pre2=-1/(mb+md2)
pre0=-1/(mb+md0)
pre4=-1/(mb+md4)
pre5=-1/(mb+md5)

#nsq0=pd.read_csv('./hA1/hA1-nsq0.txt',sep=' ',header=None)
nsq1=pd.read_csv('Ratios/A0/A0-nsq1.txt', sep=' ', header=None)
nsq2=pd.read_csv('Ratios/A0/A0-nsq2.txt', sep=' ', header=None)
nsq3=pd.read_csv('Ratios/A0/A0-nsq3.txt', sep=' ', header=None)
nsq4=pd.read_csv('Ratios/A0/A0-nsq4.txt', sep=' ', header=None)
nsq5=pd.read_csv('Ratios/A0/A0-nsq5.txt', sep=' ', header=None)

#nsq0plt=pd.read_csv('./Fits/hA1-Av-nsq0-Fit.csv',sep='\s')
nsq1plt=pd.read_csv('./Fits/A0/A0-Av-nsq1-Fit.csv',sep='\s')
nsq2plt=pd.read_csv('./Fits/A0/A0-Av-nsq2-Fit.csv',sep='\s')
nsq3plt=pd.read_csv('./Fits/A0/A0-Av-nsq3-Fit.csv',sep='\s')
nsq4plt=pd.read_csv('./Fits/A0/A0-Av-nsq4-Fit.csv',sep='\s')
nsq5plt=pd.read_csv('./Fits/A0/A0-Av-nsq5-Fit.csv',sep='\s')

#x0, y0 = [-1, 27], [-nsq0plt['EffectiveMass'], -nsq0plt['EffectiveMass']]
x1, y1 = [-1, 27], [nsq1plt['EffectiveMass'], nsq1plt['EffectiveMass']]
x2, y2 = [-1, 27], [nsq2plt['EffectiveMass'], nsq2plt['EffectiveMass']]
x3, y3 = [-1, 27], [nsq3plt['EffectiveMass'], nsq3plt['EffectiveMass']]
x4, y4 = [-1, 27], [nsq4plt['EffectiveMass'], nsq4plt['EffectiveMass']]
x5, y5 = [-1, 27], [nsq5plt['EffectiveMass'], nsq5plt['EffectiveMass']]

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

figure_size = (6, 4)
plt.xlabel('Time',fontsize=15)
plt.ylabel(r'$\widetilde{A}_0$',fontsize=15)
#plt.plot(range(96),nsq1[0])
#plt.plot(range(25),avn0y[0:25])
#plt.plot(range(25),avn0z[0:25])
#plt.plot(range(25),avn0[0:25])
#plt.plot(range(96),avn00)
#plt.errorbar(list(range(25)), -pre0*nsq0[0][0:25], yerr=-pre0*nsq0[1][0:25],ls='none',fmt='x',label='$n^2=0$',color='g')

plt.errorbar(list(range(25)), nsq1[0][0:25], yerr=nsq1[1][0:25],ls='none',fmt='x',label='$n^2=1$',color='b')
plt.errorbar(list(range(25)), nsq2[0][0:25], yerr=nsq2[1][0:25],ls='none',fmt='x',label='$n^2=2$',color='orange')
plt.errorbar(list(range(25)), nsq3[0][0:25], yerr=nsq3[1][0:25],ls='none',fmt='x',label='$n^2=3$',color='brown')
plt.errorbar(list(range(25)), nsq4[0][0:25], yerr=nsq4[1][0:25],ls='none',fmt='x',label='$n^2=4$',color='red')
plt.errorbar(list(range(25)), nsq5[0][0:25], yerr=nsq5[1][0:25],ls='none',fmt='x',label='$n^2=5$',color='magenta')

#plt.plot(x0,y0,color='g')
#plt.fill_between(list(range(47))[int(reg_low0):int(reg_up0+1)], -nsq0plt['EffectiveMass']+sigma0, -nsq0plt['EffectiveMass']-sigma0, color='g',alpha=0.2)

plt.plot(x1,y1, color='b',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low1):int(reg_up1+1)], nsq1plt['EffectiveMass']+sigma1, nsq1plt['EffectiveMass']-sigma1, color='b',alpha=0.2)
plt.plot(x2,y2,color='orange',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low2):int(reg_up2+1)], nsq2plt['EffectiveMass']+sigma2, nsq2plt['EffectiveMass']-sigma2, color='orange',alpha=0.2)
plt.plot(x3,y3, color='brown',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low1):int(reg_up1+1)], nsq3plt['EffectiveMass']+sigma3, nsq3plt['EffectiveMass']-sigma3, color='brown',alpha=0.2)
plt.plot(x4,y4, color='red',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low4):int(reg_up4+1)], nsq4plt['EffectiveMass']+sigma4, nsq4plt['EffectiveMass']-sigma4, color='red',alpha=0.2)
plt.plot(x5,y5, color='magenta',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low5):int(reg_up5+1)], nsq5plt['EffectiveMass']+sigma5, nsq5plt['EffectiveMass']-sigma5, color='magenta',alpha=0.2)

plt.annotate(r'$\bf{preliminary}$',xy=(0.17,0.03),xycoords='axes fraction',fontsize=15,color='grey',alpha=.7)
#plt.axis((0,25,0,4))
plt.tick_params(axis='both', which='major', labelsize=14)  # For ma


#plt.yscale('log')
plt.legend()
plt.savefig('Niceplot-A0.pdf',transparent=True,dpi=250,bbox_inches='tight')
'''