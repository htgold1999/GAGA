# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:44:08 2022

@author: Hannah Gold
"""



# %%


#### RUN THIS FILE IN ORDER TO START GENETIC ALGORITHM #############################################################


from Genetic_algo import Genetic_algo
from Epsilon_Creator import Epsilon_Creator
from Magnetooptic2Magnetooptic import *
from magnetophotonicCrystal import magnetophotonicCrystal
from Data import *
from Data_frame_creator import material_dict_df



# Physical constants.
hbar = 1.05457e-34
eV = 1.60218e-19
c = 2.99792e8
k_B = 1.38065e-23
sigma = 5.67037e-8
m_e = 9.10938e-31
epsilon_0 = 8.85419e-12


# TOTAL ITERATIONS
n_iter = 30   
# bits
n_bits = 6
# SIZE OF POPULATION
n_pop = 40 
# CROSSOVER RATE
r_cross = 0.9
# MUTATION RATE
r_mut = 1.0 / float(n_bits)
# G FLIP PROBABILITY
g_flip= 0.5



lambda_=np.linspace(10*(10**-6),2.25*(10**-5),600)

unit_cells=[[1,3]]

x=Epsilon_Creator()
material_dictionary=x.material_dict()
y=Genetic_algo(lambda_,[10],['p'],np.linspace(60,450,400)*(10**-9),unit_cell_pairs=5,unit_cell=unit_cells,g_flip=g_flip, material_dict_=material_dictionary) 
best_design, score, score_tracker,best_vert_dist=y.genetic_algorithm(n_bits, n_iter, n_pop, r_cross, r_mut)

print('GENETIC ALGORITHM COMPLETED!')
print('best design: %s \n score = %f' % (best_design, score)) 



# %% 
