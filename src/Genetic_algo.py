# -*- coding: utf-8 -*-
"""
Created on Wed October 27 13:34:44 2022

@author: Hannah Gold
"""
import numpy as np
from numpy.random import randint
from numpy.random import rand
from numpy.linalg import norm
import os
import scipy.stats as sc
import scipy.signal as ss
import time

import copy

from Epsilon_Creator import Epsilon_Creator
from Magnetooptic2Magnetooptic import *
from magnetophotonicCrystal import magnetophotonicCrystal
from Data import *
from Data_frame_creator import material_dict_df

from tqdm import trange 
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

      # Physical constants.
pi = np.pi
hbar = 1.05457e-34
eV = 1.60218e-19
c = 2.99792e8
k_B = 1.38065e-23
sigma = 5.67037e-8
m_e = 9.10938e-31
epsilon_0 = 8.85419e-12




import random
class Genetic_algo():
    def __init__(self,lambda_,layers=[],pol=[],thick=[],unit_cell_pairs=None,unit_cell=None,g_flip=0, material_dict_=None):
        #def __init__(self,omega,materials,layers=[],pol=[],thetas=[],thick=[]):
    #other possible parameters:b_field
        #self.omega=omega   #omega in rad/s
        #self.thetas=thetas #list of possible thetas
        #self.eps_obj=epsilon_creator_obj # contains epsilon creator object from run.py
        self.material_dict_=material_dict_
        self.unit_cell_pairs=unit_cell_pairs
        self.unit_cell=unit_cell
        self.g_flip=g_flip
        self.lambda_=lambda_ # array of wavelengths
        self.layers=layers #list of possible number of layers
        self.thick=thick #list of possible thicknesses
        self.pol=pol #list of polarizations
        

    def population(self, pop_n):
        print("start")
        pop=[]
        
        for i in range(0,pop_n):
            list0=[]
            list1=[]
            
            numb_layers=random.choice(self.layers)
            list1.append(numb_layers)
            list1.append(random.choice(self.pol))
            list0.append(list1)
           
            list_=[]
            list__=[]
            if self.unit_cell!=None:
                thickness=[]
                
                for xii in range(0,self.unit_cell_pairs):  
                  
                    
                    material=[]
                    
                    unit_material_pair=[]
                    unit_cell=random.choice(self.unit_cell)
                    for jj in range(0,len(unit_cell)):
                        
                        material=(random.choice(list(self.material_dict_[unit_cell[jj]].keys())) )
                        
                        
                        thickness=(random.choice(self.thick))
                        list__.append(thickness)
                        if rand() > self.g_flip:
                            g="g"
                        else:
                            g="-g"
                        unit_material_pair.append((unit_cell[jj],material, g))
                       
                    
                    list_.append(unit_material_pair)
                    
                    
                list__.pop(0)#pop one of the thicknesses since the last layer is semi-infinite
                list0.append(list_)
                list0.append(list__)
                pop.append(list0)
            
                
           
        return pop
        
 
    def objective(self, x):
        
        output=0
     
        pol=x[0][1]
        
        N=x[0][0]
       
        
        theta = np.array([-55,55])
        theta = theta*(pi/180)  # Angle of incidence [rad].
     
        material_dict_indices=x[1]
      
        t=x[2]
      
        omega=2*pi*c/self.lambda_
        
        Rp=np.empty((len(theta),len(self.lambda_)))
        Tp=np.empty((len(theta),len(self.lambda_)))
        Rs=np.empty((len(theta),len(self.lambda_)))
        Ts=np.empty((len(theta),len(self.lambda_)))
        for ii in range(0,len(theta)):
            for jj in range(0,len(self.lambda_)):
                z=Epsilon_Creator(dataframe=material_dict_df,lambda__=(self.lambda_[jj]),  N_=N,  material_dict_indices_=material_dict_indices)
                epsilon=z.create_epsilon()
                Rp[ii][jj], Tp[ii][jj]=magnetophotonicCrystal(self.lambda_[jj], theta[ii], N, epsilon, t, 'p')
                Rs[ii][jj], Ts[ii][jj]=magnetophotonicCrystal(self.lambda_[jj], theta[ii], N, epsilon, t, 's')
              
                
        Ap=1-(Rp+Tp)
        As=1-(Rs+Ts)
        energy=hbar*omega/eV
            
        
        contrast =abs(Ap[0]-Ap[1])
        maxcontrast =max(contrast)
        idx=np.where(contrast==maxcontrast)
        
        if Ap[0][idx]> Ap[1][idx]:
            FOM = (Ap[0][idx]+As[0][idx])/(Ap[1][idx]+As[1][idx])
        else: 
            FOM = (Ap[1][idx]+As[1][idx])/(Ap[0][idx]+As[0][idx])
        
           
        output =(FOM)
        # output=maxcontrast 
        
        return output,maxcontrast, x


   

    def selection(self, pop, scores, k=3):
      # print("inside selection")
      #first random selection
      selection_ix=randint(len(pop))
      for ix in randint(0,len(pop), k-1):
        if scores[ix] > scores[selection_ix]:
          selection_ix = ix
      return pop[selection_ix]
  

    
    def crossover(self,p1, p2, r_cross):
        c1, c2 = p1.copy(), p2.copy()
        
        
        if len(p1[1])>3:
            if rand() < r_cross:
                list11=p1[1] #materials
                list21=p1[2] #thicknesses
                
                list12=p2[1] #materials
                list22=p2[2] #thicknesses
                
                # select crossover point that is not on the end of the string
                pt = randint(1, len(list11)-2) #subtract 2 so that the individuals will not be split at the ends
                # time for crossover
                c1 = [p1[0]]+[list11[:pt] + list12[pt:]]+[list21[:pt]+list22[pt:]]
                c2 = [p2[0]]+[list12[:pt] + list11[pt:]]+[list22[:pt]+list21[pt:]]
                
        
        else: 
            c1, c2 = p1, p2
        return [c1, c2]
     

    def mutation(self,attributes, r_mut):
       
        for i in range(0, len(attributes)):
            if i==0:
                if rand() < r_mut:
                    attributes[0][0] = random.choice(self.layers)  #random layer amount
                if rand() < r_mut:
                    attributes[0][1] = random.choice(self.pol) #random polarization
            elif i==1:
                # print("you are here")
                for ii in range(0, len(attributes[i])-1):
                    if rand() < r_mut:
                  
                        unit_material_pair=[]
                        
                        unit_cell=random.choice(self.unit_cell)
                        for jj in range(0,len(unit_cell)):
                            
                            
                            material=(random.choice(list(self.material_dict_[unit_cell[jj]].keys())) )
                            if rand() > self.g_flip:
                                g="g"
                            else:
                                g="-g"
                            unit_material_pair.append((unit_cell[jj],material, g))
                        attributes[i][ii]=unit_material_pair
                        
            elif i==2:
                for kk in range(0, len(attributes[i])-1):
                    if rand() < r_mut:
                            print('________THICKNESS MUTATION______________')
                            attributes[i][ii]=random.choice(self.thick)
        return attributes
                            
 ###########################################################################################                           
                            
    def genetic_algorithm(self, n_bits, n_iter, n_pop, r_cross, r_mut):
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        
        print("inside genetic algorithm")
        pop=self.population(n_pop)
        #print('pop')
        #print(pop)
        # keep track of best solution
        best_eval,L1,L2 =  self.objective(pop[0]) #objective needs to take in inputs of the format we have
        best = pop[0]
        print("best")
        print(best)
        print("best_eval")
        print(best_eval)
     
        counter_iter=0
        score_tracker=[]
        best_vert_dist =0
        for gen in trange(n_iter):
        
        # while best_vert_dist<=0.890:
        
    
            counter_iter=counter_iter+1
            
            #print("Iterated %d/%d times" % (counter_iter,n_iter))
            # evaluate all candidates in the population
            outputs = [self.objective(c) for c in pop]
            # print("outputs-----------------")
            # print(outputs)
            scores=[xx[0] for xx in outputs]
            # print("scores")
            # print(scores)
            score_tracker.append(sum(scores)/len(scores))
            # print("scores")
            # print(scores)
            # check for new best solution
            
            for i in range(n_pop):
                score=outputs[i][0]
                if score> best_eval:
                   
                    best_vert_dist = outputs[i][1]
                    best_eval=copy.deepcopy(outputs[i][0])
                    best = copy.deepcopy(outputs[i][2])
                    print(best,best_vert_dist)
                    print(" new best f(%s) = %.3f" % ( pop[i], outputs[i][0]))
                    with open('Iterations/'+str(timestr)+'.txt', 'a') as f:
                        f.write("\n")
                        f.write(str(best))
                        f.write("\n")
                        f.write("\n")
                        f.write('best_eval='+str(best_eval)+","+'vertical_dist='+str(L1))
                        f.write("\n")
                        
            # get the best parents
            selected = [self.selection(pop, scores) for _ in range(n_pop)]
            # make newer better population
            children = list()
            for i in range(0, n_pop, 2):
                # parents two at a time
                p1, p2 = selected[i], selected[i+1]
                # crossover and mutation
                #print("entering crossover")
                for c in self.crossover(p1, p2, r_cross):
                    # mutation
                    c_new=self.mutation(c, r_mut)
                    # store for next generation
                    children.append(c_new)
            # replace population
            pop = children
        return [best, best_eval, score_tracker, best_vert_dist]

            
