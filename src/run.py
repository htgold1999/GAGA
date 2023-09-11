# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:44:08 2022

@author: Hannah Gold
"""
import numpy as np

import os
# %%



from math import *
####RUN THIS FILE IN ORDER TO RUN THINGS IN THE CORRECT ORDER#############################################################

#NOTE: specific wavelength ranges for lambda do not work because omega gets too large
#First start by creating the data frame of the refractive indices of each material based on wavelength
#n_pop must be even
#N should be greater than 3 to allow for crossover
#df=Data_frame_creator()

# current_working_directory = os.getcwd()
# print("this path",current_working_directory)
# path =current_working_directory+'/src'
# os.chdir(path)
# current_working_directory = os.getcwd()
# print("this OTHERR path",current_working_directory)


# current_working_directory = os.getcwd()
# path =current_working_directory +'/src/Find_reflectance'
# print("this path in run.py",path)
# os.chdir(path)

current_working_directory = os.getcwd()
print("hereeee",current_working_directory)
from Genetic_algo import Genetic_algo
from Find_reflectance.Epsilon_Creator import Epsilon_Creator
from Find_reflectance.Magnetooptic2Magnetooptic import *
from Find_reflectance.magnetophotonicCrystal import magnetophotonicCrystal
from Find_reflectance.Refractive_index_data import *
from Data_frame_creator import material_dict_df


#if there are issues running look at the file path for Data_frame_creator in its script

# Physical constants.
hbar = 1.05457e-34
eV = 1.60218e-19
c = 2.99792e8
k_B = 1.38065e-23
sigma = 5.67037e-8
m_e = 9.10938e-31
epsilon_0 = 8.85419e-12


# define the total iterations
n_iter = 30   #was 25 #good one is 50
# bits
n_bits = 6#was 6
# define the population size
n_pop = 20 #was 16 #must be even #was 20
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)
# g flip probability
g_flip= 0.5

# perform the genetic algorithm search

lambda_ = np.linspace(300,400,100)*(10**-9)# larger wavelengths work better for calculation since at small wavelengths, epsilon is small and causes infinite values
lambda_ = np.linspace(900,1000,100)*(10**-9)
#lambda_ = np.linspace(700,750,100)*(10**-9)
data = np.genfromtxt('bulk_eps_ef60meV_fine.dat')#, skip_header=1,skip_footer=1,names=True,dtype=None, delimiter=' ')  # Dielectric tensor of CMG.

E = data[:,0]  # Energy [eV].
E_F = 0.06  # Fermi energy [eV].
omega = eV*E/hbar  # Angular frequency [rad/s].
k = omega/c  # Wavevector in air [1/m].
lambda_ = 2*pi/k 
lambda_=np.linspace(10*(10**-6),2.25*(10**-5),600)
#unit_cells=[[2,1   ],[2,3],[1,3]]
#unit_cells=[[1,4],[2,4]]

###Mostly use these unitcells
# material_pairings=[[1,4],[1,5],[2,4],[2,5],[3,4],[3,5]]
unit_cells=[[1,5],[2,5]]
# unit_cells=[[1,5]]
x=Epsilon_Creator()
material_dictionary=x.material_dict()
y=Genetic_algo(lambda_,[10],['p'],np.linspace(60,450,400)*(10**-9),unit_cell_pairs=5,unit_cell=unit_cells,g_flip=g_flip, material_dict_=material_dictionary) ## Ag has debugging code
best, score, score_tracker,best_vert_dist=y.genetic_algorithm(n_bits, n_iter, n_pop, r_cross, r_mut)

print('Done with Genetic Algorithm!')
print('f(%s) = %f' % (best, score)) 


#TO DO as of 11/21/2022
#make sure that the arguments make sense for genetic algo and get rid of material_indices if needed
#fix epsilon creator to match new material dictionary
#make it so that the number of layers is always even and we dont have to put unit cell number AND layer number, seems repetitive 
#genetic algo line 83-101 has unit_cell_pair--->make conventions generalizable to triplets of unit cells 
#fix in epsilon creator ValueError("Number of layers 'N' must match length of'material_dict_indices' ")

#TO DOk
#make sure all versions have file data where n^2=epsilon
#make sure dataframe is loaded correctly (I dont think this is the case in the main-- this is the debugger branch)
#check if omegas in both versions are in the correct places

#make the reflectance look like the one in bragg mirror (not good right now)
#figure out if you need to normalize the objective function
# %% 
