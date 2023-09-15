# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 08:52:47 2022

@author: Hannah Gold
"""
#  %%
#THIS IS GRADIENT ASCENT 
import numpy as np
from numpy.random import randint
from numpy.random import rand
from numpy.linalg import norm
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

from Epsilon_Creator import Epsilon_Creator
from Magnetooptic2Magnetooptic import *
from magnetophotonicCrystal import magnetophotonicCrystal
from Data import *
from Data_frame_creator import material_dict_df

from tqdm import trange 
import copy
      # Physical constants.
hbar = 1.05457e-34
eV = 1.60218e-19
c = 2.99792e8
k_B = 1.38065e-23
sigma = 5.67037e-8
m_e = 9.10938e-31
epsilon_0 = 8.85419e-12

def objective(x_,lambda_):
        output=0
        
        N=x_[0][0]

        
        theta = np.array([-55,55])
        theta = theta*(pi/180)  # Angle of incidence [rad].
        #material_dict_indices=list(list(zip(*x))[0])
        material_dict_indices=x_[1]
        #material_dict_indices.insert(0,[(0,"Air","g")])#remove when bug is gone
        t=x_[2]
        #t.insert(0,1e-9)#remove when bug is gone
        omega=2*pi*c/lambda_
        
        Rp=np.empty((len(theta),len(lambda_)))
        Tp=np.empty((len(theta),len(lambda_)))
        Rs=np.empty((len(theta),len(lambda_)))
        Ts=np.empty((len(theta),len(lambda_)))
        for ii in range(0,len(theta)):
            for jj in range(0,len(lambda_)):
                y=Epsilon_Creator(dataframe=material_dict_df,lambda__=(lambda_[jj]),  N_=N,  material_dict_indices_=material_dict_indices)
                epsilon=y.create_epsilon()
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
        return output
    
def gradient_calc(x_calc,index,delta_x,lambda_,learn_rate,score):
    old_thickness=x_calc[2][index]
    x_calc[2][index]= old_thickness + delta_x
    score_compare=objective(x_calc,lambda_)
    print("score compare", score_compare)
    
    if (score_compare-score)!=0:
        grad=(score_compare-score)/delta_x
        print("objective(x_k,lambda_)",objective(x_calc,lambda_))
        new_thickness=old_thickness+ learn_rate* (grad/ norm(grad))
        if (new_thickness >= 0) : # must have a nonnegative thickness
                    
                    
                    # x_k[2][kk]=new_thickness
                    x_calc[2].pop(index)
                    x_calc[2].insert(index,new_thickness)
                    
    return x_calc


def Grad_ascent(x, epochs, delta_x,learn_rate,lambda_, min_=60*(10^-9),max_=200*(10^-9)):
    
    score=objective(x,lambda_)
    best_score=0
    
    print("score -here",score)
    score_tracker=[]
    score_tracker.append(score)
    best_score=score
    best_eval =copy.deepcopy(x)
    
    x_k=copy.deepcopy(x)
    # x_k=x[:]
    for nn in trange(0, epochs):
        
        for kk in range(0, len(x_k[2])-1):
            x_k= gradient_calc(x_k,kk,delta_x,lambda_,learn_rate,score)
           
            score=objective(x_k,lambda_)
            final_score=score
            print("Best score",best_score)
            if score>best_score:
                best_score=score
                print("best score", best_score)
                best_eval=copy.deepcopy(x_k)
                print("best_eval",best_eval)
            else:
                print("Inside Else")
                pass
            
                    
            score_tracker.append(score)
        
    
    return best_eval, best_score, final_score, score_tracker



lambda_=np.linspace(10*(10**-6),3.5*(10**-5),300)
delta_x=2*(10**-9) 
learn_rate = 1.25*(10**-9) 
epochs=50
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [2.0115577889448518e-10, 1.3627638190954785e-07, 8.350451226281431e-08, 2.1837668341708543e-07, 1.482226482412063e-07, 3.2249748743718693e-07, 2.3969849246232793e-09, 3.890000000000033e-07, 1.0261306532663317e-07]]

x_best, best_score, final_score, score_tracker=Grad_ascent(x,epochs, delta_x,learn_rate,lambda_, min_=60*(10**-9),max_=400*(10**-9), )
plt.plot(score_tracker)
plt.xlabel("Iterations")
plt.ylabel(r"$\Delta\alpha$")
# %%



