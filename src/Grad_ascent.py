# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 08:52:47 2022

@author: Hannah Gold
"""
#  %%
#THIS IS GRADIENT ASCENT NOT DESCENT
import numpy as np
from numpy.random import randint
from numpy.random import rand
from numpy.linalg import norm
import matplotlib.pyplot as plt
import os, sys

import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

from Find_reflectance.Epsilon_Creator import Epsilon_Creator
from Find_reflectance.Magnetooptic2Magnetooptic import *
from Find_reflectance.magnetophotonicCrystal import magnetophotonicCrystal
from Find_reflectance.Refractive_index_data import *
from Data_frame_creator import material_dict_df

from tqdm import trange 
import copy

from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
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
0 #225 best
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [1.6623115577889447e-07, 1.1527638190954777e-07, 1.305025125628141e-07, 1.8933668341708548e-07, 1.6482412060301512e-07, 1.3949748743718596e-07, 9.339698492462313e-08, 2.0000000000000002e-07, 1.2661306532663317e-07]] 
#____________________
#initial 
x= [[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [1.6623115577889447e-07, 1.3527638190954777e-07, 1.205025125628141e-07, 1.8733668341708548e-07, 1.6482412060301512e-07, 1.3949748743718596e-07, 9.939698492462313e-08, 2.0000000000000002e-07, 1.2261306532663317e-07]] 
# x=[[10, 'p'], [[(1, 'SiO2', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', '-g')], [(1, 'TiO2', 'g'), (5, 'weyl_material_2', '-g')], [(2, 'MgO', '-g'), (5, 'weyl_material_2', '-g')]], [6.93734335839599e-08, 1.937844611528822e-07, 8.55639097744361e-08, 2.3213032581453633e-07, 3.5994987468671683e-07, 3.838095238095238e-07, 3.2671679197994984e-07, 3.9233082706766915e-07, 1.256140350877193e-07]] 
#new initial
x=[[10, 'p'], [[(2, 'MgO', '-g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', '-g'), (5, 'weyl_material_1', 'g')], [(2, 'MgO', '-g'), (5, 'weyl_material_2', '-g')], [(1, 'SiO2', 'g'), (5, 'weyl_material_2', 'g')]], [3.224561403508772e-07, 1.3072451704260653e-07, 2.042788050125313282e-07, 2.0711779448621555e-07, 3.6087200501253134e-07, 2.4406015037593984e-07, 1.1704286415538849e-07, 6.219799498746867e-08, 2.5543859649122806e-07]] 
 

# #after 140 iterations
# x =[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [4.223115577889455e-08, 1.1927638190954784e-07, 1.0450251256281406e-07, 1.593366834170856e-07, -1.1517587939698492e-07, 1.8749748743718575e-07, 1.1139698492462316e-07, 2.4399999999999986e-07, 1.2261306532663317e-07]]
#meaningless tester--good for seeing if x_best is actually best
# x =[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'TiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [6.02311557788954e-08, 1.062763819095479e-07, 1.5550251256281394e-07, 1.2433668341708575e-07, 6.282412060301605e-08, 2.1849748743718562e-07, 6.23969849246234e-08, 3.010000000000012e-07, 1.2261306532663317e-07]]



#---> made initial better --x=[[8, 'p'], [ (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [ 1.0727638190954789e-07, 1.3850251256281402e-07, 1.4133668341708567e-07, 2.004974874371857e-07, 7.639698492462323e-08, 2.810000000000008e-07, 1.2261306532663317e-07]]
# x=[[8, 'p'], [ [(5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [ 1.0727638190954789e-07, 1.3850251256281402e-07, 1.4133668341708567e-07, 2.004974874371857e-07, 7.639698492462323e-08, 2.810000000000008e-07, 1.2261306532663317e-07]]

###post changing MgO

x=[[9, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [ (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [6.223115577889454e-08, 1.1927638190954784e-07, 1.1650251256281408e-07, 1.553366834170856e-07, 1.794974874371858e-07, 1.0739698492462314e-07, 8.00000000000006e-07, 1.2261306532663317e-07]]
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'),(5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]],  [6.223115577889454e-08, 1.1927638190954784e-07, 1.1650251256281408e-07, 1.553366834170856e-07,1.8092964824120607e-07, 1.794974874371858e-07, 1.0739698492462314e-07, 8.00000000000006e-07, 1.2261306532663317e-07]]


#example 82 initial 

x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [2.3115577889448518e-07, 1.0027638190954785e-07, 6.650251256281431e-08, 2.0733668341708543e-07, 1.249666482412063e-07, 3.2149748743718693e-07, 2.3969849246232793e-07, 3.79600000000033e-07, 1.2261306532663317e-07]]

#best with learn_rate = 1.75*(10**-9) 140 iterations
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [9.9152577889448518e-08, 1.07458190954785e-07, 6.120951256281431e-08, 2.0733668341708543e-07, 1.449298982412063e-07, 3.1740748743718693e-07, 2.3769849246232793e-09, 3.890000000000033e-07, 1.2961306532663317e-07]]
##best with learn_rate = 1*(10**-9) 250 iterations
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [9.761325777889448518e-08, 9.7238190954785e-08, 9.324951256281431e-08, 1.8503668341708543e-07, 1.249294982412063e-07, 2.9729748743718693e-07, 2.326987626232793e-09, 3.332650000000033e-07, 1.0561306532663317e-07]]









#50 iterations learn_rate = 1*(10**-9) 
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [3.3115577889448518e-10, 1.140092638190954785e-07, 6.2350251256281431e-08, 2.2733968341708543e-07, 2.263292482412063e-07, 2.4261360748743718693e-07, 7.3969849246232793e-09, 2.890000000000033e-07, 3.2261306532663317e-07]]


[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [2.3115577889448518e-10, 1.1527638190954785e-07, 6.250251256281431e-08, 1.9733668341708543e-07, 1.269296482412063e-07, 3.0149748743718693e-07, 2.3969849246232793e-09, 8.890000000000033e-07, 1.2261306532663317e-07]]
x=[[10, 'p'], [[(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(2, 'MgO', 'g'), (5, 'weyl_material_1', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_1', '-g')], [(2, 'MgO', 'g'), (5, 'weyl_material_2', 'g')], [(1, 'SiO2', '-g'), (5, 'weyl_material_2', '-g')]], [2.0115577889448518e-10, 1.3627638190954785e-07, 8.350451226281431e-08, 2.1837668341708543e-07, 1.482226482412063e-07, 3.2249748743718693e-07, 2.3969849246232793e-09, 3.890000000000033e-07, 1.0261306532663317e-07]]

x_best, best_score, final_score, score_tracker=Grad_ascent(x,epochs, delta_x,learn_rate,lambda_, min_=60*(10**-9),max_=400*(10**-9), )
plt.plot(score_tracker)
plt.xlabel("Iterations")
plt.ylabel(r"$\Delta\alpha$")
# %%



