# -*- coding: utf-8 -*-


# %%   


"""
Created on Thu Oct 27 02:00:45 2022
@author: Hannah Gold
"""
from numpy import genfromtxt
import numpy as np 
import math
from math import *
import matplotlib.pyplot as plt

import cmath #
j=(cmath.sqrt(-1))


import matplotlib

from Epsilon_Creator import Epsilon_Creator
from Magnetooptic2Magnetooptic import *
from magnetophotonicCrystal import magnetophotonicCrystal
from Data import *
from Data_frame_creator import material_dict_df




matplotlib.rcParams['font.family'] = 'Arial'



      # Physical constants.
hbar = 1.05457e-34
eV = 1.60218e-19
c = 2.99792e8
k_B = 1.38065e-23
sigma = 5.67037e-8
m_e = 9.10938e-31
epsilon_0 = 8.85419e-12



def Abs_plotter(x,lambda_,theta):
        
    N=x[0][0]

    pol=x[0][1]
    theta = np.array([-55,55])
    theta = theta*(pi/180)  # Angle of incidence [rad].
    material_dict_indices=x[1]
    t=x[2]
    omega=2*pi*c/lambda_
    
    Rp=np.empty((len(theta),len(lambda_)))
    Tp=np.empty((len(theta),len(lambda_)))
    Rs=np.empty((len(theta),len(lambda_)))
    Ts=np.empty((len(theta),len(lambda_)))
    for ii in range(0,len(theta)):
        for jj in range(0,len(lambda_)):
            x=Epsilon_Creator(dataframe=material_dict_df,lambda__=(lambda_[jj]),  N_=N,  material_dict_indices_=material_dict_indices)
            epsilon=x.create_epsilon()
           
            Rp[ii][jj], Tp[ii][jj]=magnetophotonicCrystal(lambda_[jj], theta[ii], N, epsilon, t, 'p')
            Rs[ii][jj], Ts[ii][jj]=magnetophotonicCrystal(lambda_[jj], theta[ii], N, epsilon, t, 's')
         
            
    Ap=1-(Rp+Tp)
    As=1-(Rs+Ts)
    energy=hbar*omega/eV
    
    
            
    contrast =abs(Ap[0]-Ap[1])
    maxcontrast =max(contrast)

    idx=np.where(contrast==max(contrast))
    lamb=lambda_[idx]

    # if Ap[0][idx]> Ap[1][idx]:
    #     FOM = (Ap[0][idx]+As[0][idx])/(Ap[1][idx]+As[1][idx])
    # else: 
    #     FOM = (Ap[1][idx]+As[1][idx])/(Ap[0][idx]+As[0][idx])
    FOM=maxcontrast
        
    if pol=='p':
        polarization="P-pol"
    else:
        polarization="S-pol"


    f = plt.figure()
    ax = f.add_subplot()
    ax.tick_params(axis="both", which="both", direction="in")
    plt.ylabel("Absorptance")

    plt.xlabel("Energy [eV]")
    
    plt.plot((hbar*omega/eV).T,Ap[0], color='k',label="-55 Deg p-pol",linewidth='1.5')
    plt.plot((hbar*omega/eV).T,Ap[1], 'k--', label="55 Deg p-pol")
    plt.plot((hbar*omega/eV).T,As[0], color='r',label="-55 Deg s-pol",linewidth='1.5')
    plt.plot((hbar*omega/eV).T,As[0], 'r--',label="55 Deg s-pol",linewidth='1.5')
    
    plt.legend()
    plt.ylim(0,1)
    plt.xlim(0.07,0.123)
    
    print("epsilon")
    print(epsilon)
    E=(hbar*omega/eV).T
    
    return Rp,Rs,Tp,Ap,As, FOM,lamb




lambda_=np.linspace(10*(10**-6),3.5*(10**-5),600)

omega=2*pi*c/lambda_
Energy=hbar*omega/eV


theta = np.array([-55,55])

x=[[8, 'p'], [[ (3, 'weyl_material_2', 'g')], [(1, 'MgO', 'g'), (3, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'), (3, 'weyl_material_1', '-g')], [(3, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (3, 'weyl_material_2', '-g')]], [ 1.16276381909548e-07, 6.100451226281429e-08, 2.1462668341708548e-07, 1.3447264824120643e-07, 3.0874748743718705e-07,  4.102500000000031e-07, 1.0261306532663317e-07]]

x= [[10, 'p'], [[(4, 'Si', '-g'), (5, 'InAs', 'g')], [(4, 'Si', 'g'), (5, 'InAs', '-g')], [(4, 'Si', '-g'), (5, 'InAs', 'g')], [(4, 'Si', '-g'), (5, 'InAs', 'g')], [(4, 'Si', 'g'), (5, 'InAs', 'g')]], [7.270676691729325e-08, 2.5646616541353385e-07, 7.36842105263158e-08, 1.4601503759398498e-07, 1.3917293233082705e-07, 2.5646616541353385e-07, 3.11203007518797e-07, 2.67218045112782e-07, 1.6849624060150377e-07]]


Rp,Rs,T,Ap,As,FOM,lamb=Abs_plotter(x,lambda_,theta)
omega=2*pi*c/lamb
Energy=hbar*omega/eV
#%%
