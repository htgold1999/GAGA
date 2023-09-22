
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:22:06 2022
@author: Hannah Gold
"""

# import numpy as np 
import math
from math import *
import matplotlib.pyplot as plt
import cmath #
import os, sys
sys.path.append('./')
#from Find_reflectance.Magnetooptic2Magnetooptic import *
#from Magnetooptic2Magnetooptic import *
import numpy as np
j=(cmath.sqrt(-1))

from Magnetooptic2Magnetooptic import *
# Function to calculate the reflection coefficient of an N-layer, 1D
# magnetophotonic crystal in the Voigt configuration.
# 
# omega = angular frequency [rad/s].
# theta = polar angle of incidence [rad].
# N = number of layers [1].
# epsilon = Nx3 array containing the tsverse and longitudinal dielectric tensor
#           components and the gyration vector of each layer:
# 
#           [eps_1_t  eps_1_L  g_1]           
#           [  ...      ...    ...]
#           [eps_i_t  eps_i_L  g_i]
#           [  ...      ...    ...]
#           [eps_N_t  eps_N_L  g_N]
# 
# t = (N-1)x1 array containing the thickness of each layer except the Nth one,
#     which is semi-infinite:
#     
#           [t_1 ... t_i ... t_(N-1)]
# 
# pol = polarization (p or s).
def magnetophotonicCrystal(wvl, theta, N, eps, d, pol, verbose=0):
    
    # Wavevector components in air
    k_0 = 2*np.pi/wvl
    k_x = k_0*np.sin(theta)
    
    # Initial layer
    r, t = magnetooptic2magnetooptic(wvl, theta, eps[N-2,0], eps[N-2,1], eps[N-2,2], eps[N-1,0], eps[N-1,1], eps[N-1,2], pol)
    lastGamma = r
    Theta = t
    if verbose == 1:
        print('Layer {0} (eps_t = {1}, g = {2}) to layer {3} (eps_t = {4}, g = {5})'.format(N-1, eps[N-2,0], eps[N-2,2], N, eps[N-1,0], eps[N-1,2]))
        print('r = {0}, t = {1}'.format(r, t))
        print('Gamma = {0}, Theta = {1}\n==='.format(lastGamma, Theta))
    
    for i in np.arange(N-1, 1, -1):
        if pol == "p":
            k_zi = np.sqrt((eps[i-2,0] - eps[i-2,2]**2/eps[i-2,0])*k_0**2 - k_x**2)
            k_zt = np.sqrt((eps[i-1,0] - eps[i-1,2]**2/eps[i-1,0])*k_0**2 - k_x**2)
        if pol == "s":
            k_zi = np.sqrt(eps[i-2,1]*k_0**2 - k_x**2)
            k_zt = np.sqrt(eps[i-1,1]*k_0**2 - k_x**2)
        r_it, t_it = magnetooptic2magnetooptic(wvl, theta, eps[i-2,0], eps[i-2,1], eps[i-2,2], eps[i-1,0], eps[i-1,1], eps[i-1,2], pol)
        r_ti, t_ti = magnetooptic2magnetooptic(wvl, theta, eps[i-1,0], eps[i-1,1], -eps[i-1,2], eps[i-2,0], eps[i-2,1], -eps[i-2,2], pol)
        Gamma = (r_it + (1 + r_it + r_ti)*lastGamma*np.exp(2.j*k_zt*d[i-1]))/(1 - r_ti*lastGamma*np.exp(2.j*k_zt*d[i-1]))
        Theta = Theta*(1 + r_it)*np.exp(1.j*k_zt*d[i-1])/(1 - r_ti*lastGamma*np.exp(2.j*k_zt*d[i-1]))
        
        if verbose == 1:
            print('Layer {0} (eps_t = {1}, g = {2}) to layer {3} (eps_t = {4}, g = {5})'.format(i-1, eps[i-2,0], eps[i-2,2], i, eps[i-1,0], eps[i-1,2]))
            print('r = {0}, t = {1}'.format(r_it, t_it))
            print('Gamma = {0}, Theta = {1}\n==='.format(Gamma, Theta))
        
        lastGamma = Gamma
    
    # Final layer
    k_zi = np.sqrt(k_0**2 - k_x**2)
    if pol == "p":
        k_zt = np.sqrt((eps[0,0] - eps[0,2]**2/eps[0,0])*k_0**2 - k_x**2)
    if pol == "s":
        k_zt = np.sqrt(eps[0,1]*k_0**2 - k_x**2)
    r_it, t_it = magnetooptic2magnetooptic(wvl, theta, 1, 1, 0, eps[0,0], eps[0,1], eps[0,2], pol)
    r_ti, t_ti = magnetooptic2magnetooptic(wvl, theta, eps[0,0], eps[0,1], -eps[0,2], 1, 1, 0, pol)
    Gamma = (r_it + (1 + r_it + r_ti)*lastGamma*np.exp(2.j*k_zt*d[0]))/(1 - r_ti*lastGamma*np.exp(2.j*k_zt*d[0]))
    Theta = Theta*(1 + r_it)*np.exp(1.j*k_zt*d[0])/(1 - r_ti*lastGamma*np.exp(2.j*k_zt*d[0]))
    D=(1 - r_ti*lastGamma*np.exp(2.j*k_zt*d[0]))
    # Reflectance and transmittance
    if pol == "p":
        k_z0 = np.sqrt(k_0**2 - k_x**2)
        k_zN = np.sqrt((eps[-1,0] - eps[-1,2]**2/eps[-1,0])*k_0**2 - k_x**2)
        k_zN_0 = np.sqrt(eps[-1,0]*k_0**2 - k_x**2)
        # Effective dielectric constant of the Nth layer...
        eps_eff_N = eps[-1,0]*k_zN/k_zN_0 - 1.j*eps[-1,2]*k_x/k_zN_0
        R = np.abs(Gamma)**2
        if verbose == 1:
            print('{0}'.format(eps_eff_N))
        T = np.abs(Theta)**2*np.real((eps_eff_N/k_zN_0)/(1/k_z0))
    if pol == "s":
        k_z0 = np.sqrt(k_0**2 - k_x**2)
        k_zN = np.sqrt(eps[-1,1]*k_0**2 - k_x**2)
        R = np.abs(Gamma)**2
        T = np.abs(Theta)**2*np.real(k_zN/k_z0)
    
    return R, T