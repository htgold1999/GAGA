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





# TOTAL ITERATIONS
n_iter = 30   
# SIZE OF POPULATION
n_pop = 40 # needs to be an even number
# CROSSOVER RATE
r_cross = 0.9
# MUTATION RATE
r_mut = 1.0 / 6
# G FLIP PROBABILITY
g_flip= 0.5
# g_flip= 0 # make this zero for InAs


######## MATERIAL DICTIONARY #######

# Unit cells guarantee that one material from each key are paired together 
# The current material dictionary is located in Epsilon_creator.py


# 0: Air 
# 1: TiO2,SiO2, MgO
# 2: Ag,Pt
# 3: weyl_material_1, weyl_material_2
# 4: Si 
# 5: InAs
# 6: GaAs 
# 7: InSb

##########################################


lambda_=np.linspace(10*(10**-6),2.25*(10**-5),600)
#lambda_=np.linspace(6*(10**-6),1.6*(10**-5),600)

unit_cells=[[1,3]]
# unit_cells=[[4,5]]
theta = np.array([-55,55])
x=Epsilon_Creator()
material_dictionary=x.material_dict()
y=Genetic_algo(lambda_,theta,[10],['p'],np.linspace(60,450,400)*(10**-9),unit_cell_pairs=5,unit_cell=unit_cells,g_flip=g_flip, material_dict_=material_dictionary) 
best_design, score, score_tracker,best_vert_dist=y.genetic_algorithm( n_iter, n_pop, r_cross, r_mut)

print('GENETIC ALGORITHM COMPLETED!')
print('best design: %s \n score = %f' % (best_design, score)) 



# %% 
