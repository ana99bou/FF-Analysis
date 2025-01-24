#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:15:26 2024

@author: anastasiaboushmelev
"""

import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file
import matplotlib.pyplot as plt

mb=1.9257122802734448
md0=0.73484532
md1=0.7458868408203149
md2=0.7567413345078149
pre1=md0/(2*md1*mb)
pre2=md0/(2*md2*mb)

nsq0=pd.read_csv('./Mass-Ds248-0.csv',sep='\s')
nsq1=pd.read_csv('./Mass-Ds248-1.csv',sep='\s')
nsq2=pd.read_csv('./Mass-Ds248-2.csv',sep='\s')
nsq3=pd.read_csv('./Mass-Ds248-3.csv',sep='\s')
nsq4=pd.read_csv('./Mass-Ds248-4.csv',sep='\s')
nsq5=pd.read_csv('./Mass-Ds248-5.csv',sep='\s')

nsq0plt=pd.read_csv('./Ds248Result-0.csv',sep='\s')
nsq1plt=pd.read_csv('./Ds248Result-1.csv',sep='\s')
nsq2plt=pd.read_csv('./Ds248Result-2.csv',sep='\s')
nsq3plt=pd.read_csv('./Ds248Result-3.csv',sep='\s')
nsq3plt=pd.read_csv('./Ds248Result-3.csv',sep='\s')
nsq4plt=pd.read_csv('./Ds248Result-4.csv',sep='\s')
nsq5plt=pd.read_csv('./Ds248Result-5.csv',sep='\s')

x0, y0 = [1, 46], [nsq0plt['EffectiveMass'], nsq0plt['EffectiveMass']]
x1, y1 = [1, 46], [nsq1plt['EffectiveMass'], nsq1plt['EffectiveMass']]
x2, y2 = [1, 46], [nsq2plt['EffectiveMass'], nsq2plt['EffectiveMass']]
x3, y3 = [1, 46], [nsq3plt['EffectiveMass'], nsq3plt['EffectiveMass']]
x4, y4 = [1, 46], [nsq4plt['EffectiveMass'], nsq4plt['EffectiveMass']]
x5, y5 = [1, 46], [nsq5plt['EffectiveMass'], nsq5plt['EffectiveMass']]
reg_low1=nsq1plt['RegLow']
reg_up1=nsq1plt['RegUp']
sigma1=nsq1plt['Error']
reg_low2=nsq2plt['RegLow']
reg_up2=nsq2plt['RegUp']
sigma2=nsq2plt['Error']
reg_low0=nsq0plt['RegLow']
reg_up0=nsq0plt['RegUp']
sigma0=nsq0plt['Error']
reg_low3=nsq3plt['RegLow']
reg_up3=nsq3plt['RegUp']
sigma3=nsq3plt['Error']
reg_low4=nsq4plt['RegLow']
reg_up4=nsq4plt['RegUp']
sigma4=nsq4plt['Error']
reg_low5=nsq5plt['RegLow']
reg_up5=nsq5plt['RegUp']
sigma5=nsq5plt['Error']


plt.xlabel('Time',fontsize=15)
plt.ylabel('Effective Energy',fontsize=15)
#plt.plot(range(96),nsq1[0])
#plt.plot(range(45),avn0y[0:45])
#plt.plot(range(45),avn0z[0:45])
#plt.plot(range(45),avn0[0:45])
#plt.plot(range(96),avn00)

print(nsq0['EffectiveMass'])
plt.errorbar(list(range(46)), nsq0['EffectiveMass'], yerr=nsq0['Error'],ls='none',fmt='x',label='$n^2=0$',color='g')
plt.errorbar(list(range(46)), nsq1['EffectiveMass'], yerr=nsq1['Error'],ls='none',fmt='x',label='$n^2=1$',color='b')
plt.errorbar(list(range(46)), nsq2['EffectiveMass'], yerr=nsq2['Error'],ls='none',fmt='x',label='$n^2=2$',color='orange')
plt.errorbar(list(range(46)), nsq3['EffectiveMass'], yerr=nsq3['Error'],ls='none',fmt='x',label='$n^2=3$',color='brown')
plt.errorbar(list(range(46)), nsq4['EffectiveMass'], yerr=nsq4['Error'],ls='none',fmt='x',label='$n^2=4$',color='red')
plt.errorbar(list(range(46)), nsq5['EffectiveMass'], yerr=nsq5['Error'],ls='none',fmt='x',label='$n^2=5$',color='cyan')


plt.plot(x0,y0,color='g',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low0):int(reg_up0+1)], nsq0plt['EffectiveMass']+sigma0, nsq0plt['EffectiveMass']-sigma0, color='g',alpha=0.2)
plt.annotate(r'$\bf{preliminary}$',xy=(0.65,0.03),xycoords='axes fraction',fontsize=15,color='grey',alpha=1)

plt.plot(x1,y1, color='b',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low1):int(reg_up1+1)], nsq1plt['EffectiveMass']+sigma1, nsq1plt['EffectiveMass']-sigma1, color='b',alpha=0.2)
plt.plot(x2,y2,color='orange',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low2):int(reg_up2+1)], nsq2plt['EffectiveMass']+sigma2, nsq2plt['EffectiveMass']-sigma2, color='orange',alpha=0.2)
plt.ylim(bottom=0.71)
plt.ylim(top=0.88)
plt.plot(x3,y3, color='brown',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low3):int(reg_up3+1)], nsq3plt['EffectiveMass']+sigma3, nsq3plt['EffectiveMass']-sigma3, color='brown',alpha=0.2)
plt.plot(x4,y4, color='red',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low4):int(reg_up4+1)], nsq4plt['EffectiveMass']+sigma4, nsq4plt['EffectiveMass']-sigma4, color='red',alpha=0.2)
plt.plot(x5,y5, color='cyan',linewidth=0.5)
plt.fill_between(list(range(47))[int(reg_low5):int(reg_up5+1)], nsq5plt['EffectiveMass']+sigma5, nsq5plt['EffectiveMass']-sigma5, color='cyan',alpha=0.2)

plt.tick_params(axis='both', which='major', labelsize=14)  # For major ticks

plt.axvline(x=18, color='grey', linestyle='--', linewidth=0.8, label='Fit Range')  # At x=1
plt.axvline(x=25, color='grey', linestyle='--', linewidth=0.8)  # At x=4
plt.axis((0,46,0.71,0.875))


#plt.yscale('log')
plt.legend(ncol=2)
plt.savefig('Niceplot-Ds.pdf', bbox_inches='tight')