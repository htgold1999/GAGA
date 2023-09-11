# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:40:56 2022

@author: Hannah Gold
"""

import numpy as np 
import math
from math import *
import matplotlib.pyplot as plt
import cmath #
j=(cmath.sqrt(-1)) 


def magnetooptic2magnetooptic(wvl,theta, eps_1_t, eps_1_L, g_1, eps_2_t, eps_2_L, g_2, pol):
    # Wavevector components in air
    k_0 = 2*np.pi/wvl
    k_x = k_0*np.sin(theta)
    
    if pol == "s":
        # Eigenvalues
        k_z1 = np.sqrt(eps_1_L*k_0**2 - k_x**2)
        k_z2 = np.sqrt(eps_2_L*k_0**2 - k_x**2)
        # Reflection coefficient
        r = (k_z1 - k_z2)/(k_z1 + k_z2)
    if pol == "p":
        # Eigenvalues
        eps_v1 = eps_1_t - g_1**2/eps_1_t
        k_z1 = np.sqrt(eps_v1*k_0**2 - k_x**2)
        k_z1_0 = np.sqrt(eps_1_t*k_0**2 - k_x**2)
        eps_v2 = eps_2_t - g_2**2/eps_2_t
        k_z2 = np.sqrt(eps_v2*k_0**2 - k_x**2)
        k_z2_0 = np.sqrt(eps_2_t*k_0**2 - k_x**2)
        # Reflection coefficient
        r = ((eps_1_t*k_z1 - 1.j*g_1*k_x)*k_z2_0/k_z1_0 - (eps_2_t*k_z2 - 1.j*g_2*k_x)*k_z1_0/k_z2_0)/((eps_1_t*k_z1 + 1.j*g_1*k_x)*k_z2_0/k_z1_0 + (eps_2_t*k_z2 - 1.j*g_2*k_x)*k_z1_0/k_z2_0)
    
    t = 1 + r
    
    return r, t