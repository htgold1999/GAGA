# -*- coding: utf-8 -*-


# %%   


"""
Created on Thu Oct 27 02:00:45 2022
@author: Hannah Gold
"""
from numpy import genfromtxt
# from magnetophotonicCrystal import *
import numpy as np 
import math
from math import *
import matplotlib.pyplot as plt


import cmath #
j=(cmath.sqrt(-1))

import os
import matplotlib

from Epsilon_Creator import Epsilon_Creator
from Magnetooptic2Magnetooptic import *
from magnetophotonicCrystal import magnetophotonicCrystal
from Data import *
from Data_frame_creator import material_dict_df


import ast



matplotlib.rcParams['font.family'] = 'Arial'



      # Physical constants.
hbar = 1.05457e-34
eV = 1.60218e-19
c = 2.99792e8
k_B = 1.38065e-23
sigma = 5.67037e-8
m_e = 9.10938e-31
epsilon_0 = 8.85419e-12



def Reflectivity_plotter(x,color,lambda_,theta):
        
    N=x[0][0]

    pol=x[0][1]
    theta = np.array([-55,55])
    theta = theta*(pi/180)  # Angle of incidence [rad].
    #material_dict_indices=list(list(zip(*x))[0])
    material_dict_indices=x[1]
    #material_dict_indices.insert(0,[(0,"Air","g")])#remove when bug is gone
    t=x[2]
    #t.insert(0,1e-9)#remove when bug is gone
    omega=2*pi*c/lambda_
    
    Rp=np.empty((len(theta),len(lambda_)))
    Tp=np.empty((len(theta),len(lambda_)))
    Rs=np.empty((len(theta),len(lambda_)))
    Ts=np.empty((len(theta),len(lambda_)))
    for ii in range(0,len(theta)):
        for jj in range(0,len(lambda_)):
            x=Epsilon_Creator(dataframe=material_dict_df,lambda__=(lambda_[jj]),  N_=N,  material_dict_indices_=material_dict_indices)
            epsilon=x.create_epsilon()
            # print("EPSILON")
            # print(epsilon)
            Rp[ii][jj], Tp[ii][jj]=magnetophotonicCrystal(lambda_[jj], theta[ii], N, epsilon, t, 'p')
            Rs[ii][jj], Ts[ii][jj]=magnetophotonicCrystal(lambda_[jj], theta[ii], N, epsilon, t, 's')
            #gamma=magnetophotonicCrystal(omega[jj], theta[ii], N, epsilon, t, pol)
            #R[ii][jj]=(abs(gamma)**2 )
            
    Ap=1-(Rp+Tp)
    As=1-(Rs+Ts)
    energy=hbar*omega/eV
    
    
            
    contrast =abs(Ap[0]-Ap[1])
    maxcontrast =max(contrast)
    idx=np.where(contrast==maxcontrast)

    # if Ap[0][idx]> Ap[1][idx]:
    #     FOM = (Ap[0][idx]+As[0][idx])/(Ap[1][idx]+As[1][idx])
    # else: 
    #     FOM = (Ap[1][idx]+As[1][idx])/(Ap[0][idx]+As[0][idx])
    output=maxcontrast
        
    if pol=='p':
        polarization="P-pol"
    else:
        polarization="S-pol"
    #plt.title("Absorptance Plot, "+str(polarization)+", Theta=-55,55")
    #plt.title("Absorptance Plot, Theta=-55,55")

    f = plt.figure()
    ax = f.add_subplot()
    ax.tick_params(axis="both", which="both", direction="in")
    plt.ylabel("Absorptance")
    #plt.ylabel("(Rp$^-$ +Rs$^-$)/(Rp$^+$ +Rs$^+$)")
    plt.xlabel("Energy [eV]")
    #plt.xlabel("Wavelength [m]")
  
    #plt.plot((hbar*omega/eV).T,R[0], color='b',label="-55 Deg Reflectance")
    # plt.plot((hbar*omega/eV).T,T[0], color='g',label="-55 Deg Transmittance")
    plt.plot((hbar*omega/eV).T,Ap[0], color='k',label="-55 Deg p-pol",linewidth='1.5')
    plt.plot((hbar*omega/eV).T,Ap[1], 'k--', label="55 Deg p-pol")
    plt.plot((hbar*omega/eV).T,As[0], color='r',label="-55 Deg s-pol",linewidth='1.5')
    plt.plot((hbar*omega/eV).T,As[0], 'r--',label="55 Deg s-pol",linewidth='1.5')
    # plt.plot(np.ones([len(omega),1])*hbar*2*pi*c/(10e-6*eV),np.linspace(0,1,len(omega)),'b--')
    
    # plt.plot((hbar*omega/eV).T,T[1], 'g--',label="55 Deg Transmittance")
    #plt.plot((hbar*omega/eV).T,R[1], 'k',dashes=[7, 4],label="55 Deg p-pol")
    # plt.plot((hbar*omega/eV).T,np.abs(R[1]-R[0]), color = (0,1,0),label="|R$^-$ - R$^+$|",linewidth='1.5')
    #plt.plot((hbar*omega/eV).T,np.abs(R[1]-R[0]), 'k',label="|R$^-$ - R$^+$|",linewidth='1.5')
   
    #plt.plot((hbar*omega/eV).T,(R[0]+Rs[0])/(R[1]+Rs[1]), 'k',label="(Rp$^-$ +Rs$^-$)/(Rp$^+$ +Rs$^+$)",linewidth='1.5')
   ####################################################
    
    plt.legend()
    plt.ylim(0,1)
    plt.xlim(0.07,0.6)
    
    print("epsilon")
    print(epsilon)
    E=(hbar*omega/eV).T
    return Rp,Rs,Tp,Ap,As, output


######## TEST BENCH

# lambda_ =np.linspace(500,900,1000)*(10**-9)
# data = np.genfromtxt('bulk_eps_ef60meV_fine.dat')#, skip_header=1,skip_footer=1,names=True,dtype=None, delimiter=' ')  # Dielectric tensor of CMG.
# E = data[:,0]  # Energy [eV].
# E_F = 0.06  # Fermi energy [eV].
# omega = eV*E/hbar  # Angular frequency [rad/s].
# k = omega/c  # Wavevector in air [1/m].
# lambda_ = 2*pi/k 
# theta = np.array([-55,55])
# t=[5e-9,100e-9] #remove the last thickness
# x=[[3, 'p'], [ [(3,'Pt')],[(4,'Co2MnGa')],[(2,'MgO')]], t]
# epsilon_tensor=Reflectivity_plotter(x,lambda_,theta)
### Example 7


#Example Tester

# lambda_ =np.linspace(500,1000,1000)*(10**-9)
# data = np.genfromtxt('bulk_eps_ef60meV_fine.dat')#, skip_header=1,skip_footer=1,names=True,dtype=None, delimiter=' ')  # Dielectric tensor of CMG.
# E = data[:,0]  # Energy [eV].
# E_F = 0.06  # Fermi energy [eV].
# omega = eV*E/hbar  # Angular frequency [rad/s].
# k = omega/c  # Wavevector in air [1/m].
# lambda_ = 2*pi/k 
# lambda_=np.linspace(8.25*(10**-6),5.58*(10**-5),455)


lambda_=np.linspace(10*(10**-6),2.25*(10**-5),600)

omega=2*pi*c/lambda_
Energy=hbar*omega/eV




theta = np.array([-55,55])


[[10, 'p'], [[(1, 'TiO2', 'g'), (5, 'weyl_material_2', '-g')], [(1, 'TiO2', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', 'g'), (5, 'weyl_material_1', '-g')], [(1, 'TiO2', 'g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', '-g')]], [2.417035175879394e-08, 0.6376381909547735e-07, 1.1623115577889443e-07, 0.7120603015075375e-07, 1.4999999999999997e-07, 0.6487437185929646e-07, 0.6417085427135675e-07, 2.336683417085424e-08, 0.6964824120603015e-07]]
colory=['c','g','k','b','m','r']
color=colory[0]
#theta = np.linspace(0,89.9,200)

x= [[10, 'p'], [[(1, 'TiO2', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', '-g')], [(1, 'SiO2', 'g'), (5, 'weyl_material_2', '-g')]], [9.517587939698493e-08, 1.148743718592965e-07, 8.532663316582916e-08, 1.7537688442211056e-07, 7.899497487437187e-08, 1.0010050251256283e-07, 1.3386934673366836e-07, 1.7819095477386937e-07, 7.266331658291458e-08]]


x= [[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', '-g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', 'g'), (5, 'weyl_material_1', '-g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')]], [1.1346733668341709e-07, 7.688442211055277e-08, 9.587939698492463e-08, 1.092462311557789e-07, 7.969849246231156e-08, 1.9366834170854275e-07, 1.3246231155778896e-07, 1.092462311557789e-07, 1.1135678391959801e-07]]


#GD optimized
x = [[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [1.2623115577889465e-07, 1.2727638190954775e-07, 1.4250251256281405e-07, 1.9333668341708545e-07, 1.6082412060301514e-07, 1.4349748743718594e-07, 1.2139698492462318e-07, 2.3599999999999986e-07, 1.2661306532663316e-07]]
x= [[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [4.223115577889455e-08, 1.1927638190954784e-07, 1.0450251256281406e-07, 1.593366834170856e-07, -1.1517587939698492e-07, 1.8749748743718575e-07, 1.1139698492462316e-07, 2.4399999999999986e-07, 1.2261306532663317e-07]]
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [-1.976884422110545e-08, 9.727638190954779e-08, 1.9450251256281378e-07, 7.733668341708567e-08, -3.0517587939698414e-07, 2.6549748743718545e-07, 3.339698492462307e-08, 1.6800000000000018e-07, 1.2261306532663317e-07]]
x= [[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [1.8231155778894552e-08, 1.1927638190954784e-07, 1.2450251256281408e-07, 1.3933668341708568e-07, -2.3517587939698444e-07, 1.8749748743718575e-07, 1.1939698492462317e-07, 2.3799999999999988e-07, 1.2261306532663317e-07]]

x= [[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [1.8231155778894552e-08, 1.1927638190954784e-07, 1.2450251256281408e-07, 1.3933668341708568e-07, 2.3517587939698444e-07, 1.8749748743718575e-07, 1.1939698492462317e-07, 2.3799999999999988e-07, 1.2261306532663317e-07]]
x = [[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [6.223115577889454e-08, 1.1927638190954784e-07, 1.1650251256281408e-07, 1.553366834170856e-07, 6.082412060301517e-08, 1.794974874371858e-07, 1.0739698492462314e-07, 8.00000000000006e-07, 1.2261306532663317e-07]]
# x = [[9, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [ (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [6.223115577889454e-08, 1.1927638190954784e-07, 1.1650251256281408e-07, 1.553366834170856e-07, 1.794974874371858e-07, 1.0739698492462314e-07, 8.00000000000006e-07, 1.2261306532663317e-07]]
x=[[14, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', '-g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', 'g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', 'g'), (5, 'weyl_material_2', '-g')], [(1, 'TiO2', 'g'), (5, 'weyl_material_2', 'g')]], [1.205025125628141e-07, 9.376884422110552e-08, 7.758793969849247e-08, 1.8592964824120607e-07, 7.055276381909548e-08, 1.9929648241206031e-07, 1.1628140703517588e-07, 1.0291457286432162e-07, 8.603015075376886e-08, 9.658291457286433e-08, 1.6834170854271358e-07, 1.514572864321608e-07, 1.8874371859296483e-07]]
x= [[8, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [2.3115577889448518e-10, 1.1127638190954787e-07, 1.1050251256281411e-07, 1.7433668341708553e-07+ 2.58497487437186e-07, 8.396984924623278e-09, 8.430000000000047e-07, 1.2261306532663317e-07]]
# x=[[9, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [2.3115577889448518e-10, 1.1127638190954787e-07, 1.1050251256281411e-07, 1.7433668341708553e-07, 2.58497487437186e-07, 8.396984924623278e-09, 8.430000000000047e-07, 1.2261306532663317e-07]]
#x = [[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [1.6623115577889447e-07, 1.1527638190954777e-07, 1.305025125628141e-07, 1.8933668341708548e-07, 1.6482412060301512e-07, 1.3949748743718596e-07, 9.339698492462313e-08, 2.0000000000000002e-07, 1.2661306532663317e-07]] 
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'),(5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [2.3115577889448518e-10, 1.1127638190954787e-07, 1.1050251256281411e-07, 1.7433668341708553e-07,1.8592964824120607e-07, 2.58497487437186e-07, 8.396984924623278e-09, 8.430000000000047e-07, 1.2261306532663317e-07]]
###example 80 is good
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [2.3115577889448518e-10, 1.1527638190954785e-07, 6.250251256281431e-08, 1.9733668341708543e-07, 1.269296482412063e-07, 3.0149748743718693e-07, 2.3969849246232793e-09, 8.890000000000033e-07, 1.2261306532663317e-07]]

#### example 81 initial
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'),(5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]],  [6.223115577889454e-08, 1.1927638190954784e-07, 1.1650251256281408e-07, 1.553366834170856e-07,1.8092964824120607e-07, 1.794974874371858e-07, 1.0739698492462314e-07, 8.00000000000006e-07, 1.2261306532663317e-07]]
#uses new convention
# x= [[10, 'p'], [[(1, 'TiO2', '-g'), (4, 'weyl_material_1', '-g')], [(2, 'SiO2', '-g'), (4, 'weyl_material_1', 'g')], [(2, 'SiO2', 'g'), (4, 'weyl_material_1', '-g')], [(3, 'MgO', 'g'), (4, 'weyl_material_1', '-g')], [(1, 'TiO2', 'g'), (5, 'weyl_material_2', 'g')]], [1.7537688442211056e-07, 1.0221105527638192e-07, 8.180904522613066e-08, 1.2683417085427136e-07, 7.055276381909548e-08, 1.0010050251256283e-07, 1.8381909547738696e-07, 1.6834170854271358e-07, 1.7115577889447237e-07]] 
 ######
# x=[[10, 'p'], [[(2, 'MgO', '-g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', '-g'), (5, 'weyl_material_1', 'g')], [(2, 'MgO', '-g'), (5, 'weyl_material_2', '-g')], [(1, 'SiO2', 'g'), (5, 'weyl_material_2', 'g')]], [2.864561403508764e-07, 1.2872681704260653e-07, 2.1820050125313283e-07, 2.83117794486215e-07, 3.848020050125319e-07, 2.8006015037594064e-07, 9.29448621553886e-08, 7.419799498746878e-08, 1.5543859649122806e-07]]

# x=[[10, 'p'], [[(3, ['MgO'], '-g'), (5, ['weyl_material_2'], 'g')], [(3, ['MgO'], 'g'), (4, ['weyl_material_1'], '-g')], [(3, ['MgO'], '-g'), (4, ['weyl_material_1'], 'g')], [(3, ['MgO'], '-g'), (5, ['weyl_material_2'], '-g')], [(2, ['SiO2'], 'g'), (5, ['weyl_material_2'], 'g')]], [2.864561403508764e-07, 1.2872681704260653e-07, 2.1820050125313283e-07, 2.83117794486215e-07, 3.848020050125319e-07, 2.8006015037594064e-07, 9.29448621553886e-08, 7.419799498746878e-08, 1.5543859649122806e-07]]
# x =[[3,'p'],[[(3, ["MgO"], '-g'), (4, ['weyl_material_1'], 'g')],[(3, ["MgO"], '-g')]],[0.4e-06,0.4e-06]]
# x=[[10, 'p'], [[(3, ['MgO'], 'g'), (5, ['weyl_material_2'], 'g')], [(2, ['SiO2'], '-g'), (5, ['weyl_material_2'], '-g')], [(2, ['SiO2'], 'g'), (5, ['weyl_material_2'], '-g')], [(1, ['TiO2'], '-g'), (4, ['weyl_material_1'], '-g')], [(3, ['MgO'], '-g'), (5, ['weyl_material_2'], '-g')]], [9.408521303258146e-08, 2.9944862155388466e-07, 6.000000000000001e-08, 3.403508771929825e-07, 2.7473684210526313e-07, 2.58546365914787e-07, 2.0230576441102756e-07, 3.41203007518797e-07, 2.193483709273183e-07]] 
# x= [[10, 'p'], [[(3, 'MgO', '-g'), (5, 'weyl_material_2', 'g')], [(3, 'MgO', 'g'), (4, 'weyl_material_1', '-g')], [(3, 'MgO', '-g'), (4, 'weyl_material_1', 'g')], [(3, 'MgO', '-g'), (5, 'weyl_material_2', '-g')], [(2, 'SiO2', 'g'), (5, 'weyl_material_2', 'g')]], [2.864561403508764e-07, 1.2872681704260653e-07, 2.1820050125313283e-07, 2.83117794486215e-07, 3.848020050125319e-07, 2.8006015037594064e-07, 9.29448621553886e-08, 7.419799498746878e-08, 1.5543859649122806e-07]]
# x=[[10, 'p'], [[(2, 'MgO', '-g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', '-g'), (5, 'weyl_material_1', 'g')], [(2, 'MgO', '-g'), (5, 'weyl_material_2', '-g')], [(1, 'SiO2', 'g'), (5, 'weyl_material_2', 'g')]], [2.864561403508764e-07, 1.2872681704260653e-07, 2.1820050125313283e-07, 2.83117794486215e-07, 3.848020050125319e-07, 2.8006015037594064e-07, 9.29448621553886e-08, 7.419799498746878e-08, 1.5543859649122806e-07]]
# x =[[3,'p'],[[(1, "SiO2", '-g'), (5, 'weyl_material_1', 'g')],[(1, "SiO2", '-g')]],[0.4e-06,0.4e-06]]

x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [2.0115577889448518e-10, 1.0527638190954785e-07, 3.250251256281431e-08, 1.9733668341708543e-07, 1.269296482412063e-07, 3.0149748743718693e-07, 2.3969849246232793e-09, 3.390000000000033e-07, 1.2261306532663317e-07]]

#EF=0.2ev
# x=[[8, 'p'], [[(5, 'weyl_material_2_EF_02', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1_EF_02', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1_EF_02', '-g')], [ (5, 'weyl_material_2_EF_02', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2_EF_02', '-g')]], [ 1.16276381909548e-07, 6.100451226281429e-08, 2.1462668341708548e-07, 1.3447264824120643e-07, 3.0874748743718705e-07,  4.102500000000031e-07, 1.0261306532663317e-07]]
# # #EF=0.15ev
x=[[8, 'p'], [[ (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')], [(5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [ 1.16276381909548e-07, 6.100451226281429e-08, 2.1462668341708548e-07, 1.3447264824120643e-07, 3.0874748743718705e-07,  4.102500000000031e-07, 1.0261306532663317e-07]]

# # #EF=0.1ev
# x=[[8, 'p'], [[(5, 'weyl_material_2_EF_01', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1_EF_01', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1_EF_01', '-g')], [ (5, 'weyl_material_2_EF_01', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2_EF_01', '-g')]], [ 1.16276381909548e-07, 6.100451226281429e-08, 2.1462668341708548e-07, 1.3447264824120643e-07, 3.0874748743718705e-07,  4.102500000000031e-07, 1.0261306532663317e-07]]
# # #EF=0.07ev
# x=[[8, 'p'], [[(5, 'weyl_material_2_EF_007', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1_EF_007', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1_EF_007', '-g')], [ (5, 'weyl_material_2_EF_007', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2_EF_007', '-g')]], [ 1.16276381909548e-07, 6.100451226281429e-08, 2.1462668341708548e-07, 1.3447264824120643e-07, 3.0874748743718705e-07,  4.102500000000031e-07, 1.0261306532663317e-07]]



#random


Rp,Rs,T,Ap,As,maxcontrast=Reflectivity_plotter(x,color,lambda_,theta)

# Ap0_max=max(Ap[0])
# a_idx= np.where(Ap[0]==Ap0_max)
# a_omega_at_max=omega[a_idx]
# Ap1_at_max = Ap[1][a_idx]
# As=As[0][a_idx]
# np.savetxt("3layerMgO_Weyl1_MgO_Absorp_ppol_txt.txt",np.transpose(Ap))
# np.savetxt("3layerMgO_Weyl1_MgO_Absorp_spol_txt.txt",np.transpose(As))
# np.savetxt("omega_ideal_struct_txt.txt",np.transpose(omega))


# #x=[[1,'p'],[[(4, "Co2MnGa", '-g')]],[1e-06]]

# np.savetxt("example83_10layer_Absorp_ppol_EF_txt.txt",np.transpose(Ap))
# np.savetxt("example83_10layer_Absorp_spol_EF_txt.txt",np.transpose(As))
# np.savetxt("omega_ideal_struct_txt.txt",np.transpose(omega))

# np.savetxt("example83_8layer_Absorp_ppol_txt.txt",np.transpose(Ap))
# np.savetxt("example83_8layer_Absorp_spol_txt.txt",np.transpose(As))
# np.savetxt("omega_ideal_struct_txt.txt",np.transpose(omega))



# np.savetxt("example83_8layer_Absorp_ppol_txt.txt",Ap)
# np.savetxt("example83_8layer_Absorp_spol_txt.txt",As)
# np.savetxt("omega_ideal_struct_txt.txt",omega)




# np.save("example83_10layer_Absorp_ppol_EF_02.npy",Ap)
# np.save("example83_10layer_Absorp_spol_EF_02.npy",As)
# np.save("omega_ideal_struct.npy",omega)
# %%





# Ap = np.genfromtxt("example80_10layer_Absorp_ppol_txt.txt")
# As = np.genfromtxt("example80_10layer_Absorp_spol_txt.txt")
# omega =np.genfromtxt("omega_ideal_struct_txt.txt")




# Rp_55 =np.genfromtxt("Rp_55.txt")

# plt.plot(omega,np.transpose(Ap)[0])
# plt.plot(omega,np.transpose(Ap)[1])
# plt.plot(omega,np.transpose(As)[0])

# plt.plot(np.transpose(Rp_55)[0],1-np.transpose(Rp_55)[1])
    
    

# material_dict = { 0:{"Air": {"function": self.Air, "isotropic":1}},1:{"TiO2": {"function": self.TiO2_eps, "isotropic":1}}, 2:{"SiO2": {"function":self.SiO2_eps,"isotropic":1}}, 3:{'MgO': {"function":self.MgO_eps,"isotropic":1}}, 
                        #   4:{"weyl_material_1": {"function":self.weyl_material_1_eps,"isotropic":0}},5:{"weyl_material_2": {"function":self.weyl_material_2_eps,"isotropic":0}}} #can have more information. isotropic:1 means it isotropic, isotropic:0 means it anisotropic
    
 # Interesting examples  
x = [[9, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [ (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [6.223115577889454e-08, 1.1927638190954784e-07, 1.1650251256281408e-07, 1.553366834170856e-07, 1.794974874371858e-07, 1.0739698492462314e-07, 8.00000000000006e-07, 1.2261306532663317e-07]]
x= [[8, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [ (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [ (1, 'SiO2', '-g')]], [6.223115577889454e-08, 1.1927638190954784e-07, 1.1650251256281408e-07, 1.553366834170856e-07, 1.794974874371858e-07, 1.0739698492462314e-07, 8.00000000000006e-07]]
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'),(5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [2.3115577889448518e-10, 1.1127638190954787e-07, 1.1050251256281411e-07, 1.7433668341708553e-07,1.8592964824120607e-07, 2.58497487437186e-07, 8.396984924623278e-09, 8.430000000000047e-07, 1.2261306532663317e-07]]
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'),(5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]],  [6.223115577889454e-08, 1.1927638190954784e-07, 1.1650251256281408e-07, 1.553366834170856e-07,1.8592964824120607e-07, 1.794974874371858e-07, 1.0739698492462314e-07, 8.00000000000006e-07, 1.2261306532663317e-07]]

