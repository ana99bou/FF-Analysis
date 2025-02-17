#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:24:34 2025

@author: anastasiaboushmelev
"""
#TODO Check deltaT
def getEns(s):
    if s=='F1S': return 98,30,96,48
    elif s=='M2': return 889,26,64,32
    elif s=='C1': return 1636,20,64,24
    elif s=='C2': return 1419,20,64,24
    
def getSM(s):
    if s=='F1S': return 12.14
    elif s=='C1': return 7.86
    elif s=='M2': return 10.36

def getSmass(s):
    if s=='F1S': return 0.02144
    elif s=='C1': return 0.03224
    elif s=='M2': return 0.025
    
    
# Giving charm masses for each ensemble
def getCmass(s):
    if s=='F1S': return [0.248,0.259,0.275]
    elif s=='C1': return ['0.300','0.350','0.400']
    elif s=='M2': return ['0.280','0.3203','0.340']

#Gives m,csw,zeta
def getRHQparams(s):
    if s=='M2':return [3.49,3.07,1.76]
    elif s=='C1':return [7.47,4.92,2.93]