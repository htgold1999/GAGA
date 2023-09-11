# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:34:44 2022

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
# sys.path.append('./')

# current_working_directory = os.getcwd()
# path =current_working_directory 
# print("this path in Genetic Algo",path)
# # os.chdir(path)



# current_working_directory = os.getcwd()
# path =current_working_directory +'/Find_reflectance'
# print("this path in GAAAA",path)


#os.chdir(path)

# from Find_reflectance.Epsilon_Creator import Epsilon_Creator
# from Find_reflectance.Magnetooptic2Magnetooptic import *
# from Find_reflectance.magnetophotonicCrystal import magnetophotonicCrystal
# from Find_reflectance.Refractive_index_data import *
# from Find_reflectance.Data_frame_creator import material_dict_df

# from Epsilon_Creator import Epsilon_Creator
# from Magnetooptic2Magnetooptic import *
# from magnetophotonicCrystal import magnetophotonicCrystal
# from Refractive_index_data import *
# from Data_frame_creator import material_dict_df


from Find_reflectance.Epsilon_Creator import Epsilon_Creator
from Find_reflectance.Magnetooptic2Magnetooptic import *
from Find_reflectance.magnetophotonicCrystal import magnetophotonicCrystal
from Find_reflectance.Refractive_index_data import *
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

#genetic algorithm Python tutorial I used https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/




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
    #[[omega,theta,layers,'pol'],[material,material...],...[thickness, thickness...]] #thicknesses correspond to indices of material list
    #might make default polarizaition 'p' or just not include 's' in the list
        pop=[]
        #poppers=[]
        for i in range(0,pop_n):
            list0=[]
            list1=[]
            #list1.append(self.omega)
            #list1.append(random.choice(self.thetas))
            numb_layers=random.choice(self.layers)
            list1.append(numb_layers)
            list1.append(random.choice(self.pol))
            list0.append(list1)
            # like=[]
            list_=[]
            list__=[]
            if self.unit_cell!=None:
                ## mess with the materials list thing
                thickness=[]
                
                for xii in range(0,self.unit_cell_pairs):  
                    # mat=[]
                    # what=[]
                    
                    material=[]
                    
                    unit_material_pair=[]
                    #material=random.choice(range(1,self.materials+1)) #find a way to make material pairings, maybe a pair is just used as 1 material? two diff thicknesses required tho
                    unit_cell=random.choice(self.unit_cell)
                    for jj in range(0,len(unit_cell)):
                        # mat=((1,9))
                        # what.append(mat)
                        
                        #unit_cell_index0=unit_cell[0]
                        #unit_cell_index1=unit_cell[1]   
                    
                        # print("unit_cell[jj]")
                        # print(unit_cell[jj])
                        # print("unit_cell_index1")
                        # print(unit_cell_index0)
                        # print("self.material_dict_[unit_cell[jj]]")
                        # print(self.material_dict_[unit_cell[jj]].keys())
                        
                        material=(random.choice(list(self.material_dict_[unit_cell[jj]].keys())) )
                        #material1=random.choice(list(self.material_dict_[unit_cell_index1].keys()))
                        
                        thickness=(random.choice(self.thick))
                        list__.append(thickness)
                        if rand() > self.g_flip:
                            g="g"
                        else:
                            g="-g"
                        unit_material_pair.append((unit_cell[jj],material, g))
                        # if count!= (len(self.unit_cell_pairs))-1:
                        #     list__.append(thickness[jj]) # the last layer is semi-infinite, therefore has no specified thickness
                    #thickness1=random.choice(self.thick)
                    
                    #like.append(what)
                    
                    list_.append(unit_material_pair)
                    #poppers.append(like)
                    #list_.append((unit_cell_index1,material1))
                    #list__.append(thickness)
                    
                list__.pop(0)#pop one of the thicknesses since the last layer is semi-infinite
                list0.append(list_)
                list0.append(list__)
                pop.append(list0)
            
                
            #print("HERE IS THE POPULATION")
            #print(pop)
        return pop
        #print(pop)
 
    
    
    def objective__(self, x):
        
        
        # Define the Gaussian function
        def Gauss(x, A, B):
            y = 1-A*np.exp(-1*B*x**2)
            return y
        def Gauss0(x, mean, sigma):
           return 1-((1/(sigma*math.sqrt(2*pi)))*np.exp(-0.5*np.square((x-mean)/sigma)))
        
        #This code for "p" polarization calculation 
        pol=x[0][1]
        
        N=x[0][0]
        N=N+1 #remove when bug is gone
        
        theta = np.array([-55,55])
        theta = theta*(pi/180)  # Angle of incidence [rad].
        #material_dict_indices=list(list(zip(*x))[0])
        material_dict_indices=x[1]
        #material_dict_indices.insert(0,[(0,"Air","g")])#remove when bug is gone
        t=x[2]
        #t.insert(0,1e-9)#remove when bug is gone
        omega=2*pi*c/self.lambda_
        
        R=np.empty((len(theta),len(self.lambda_)))
        for ii in range(0,len(theta)):
            for jj in range(0,len(self.lambda_)):
                x=Epsilon_Creator(dataframe=material_dict_df,lambda__=(self.lambda_[jj]),  N_=N,  material_dict_indices_=material_dict_indices)
                epsilon=x.create_epsilon()
                gamma=magnetophotonicCrystal(omega[jj], theta[ii], N, epsilon, t, pol)
                R[ii][jj]=(abs(gamma)**2 )
        
        R_0_peaks=ss.find_peaks(-1*R[0]) ##multiplied by -1 to find toughs
        
        R_1_peaks=ss.find_peaks(-1*R[1])
        print(" R_1_peaks")
        print( R_1_peaks)
        print("test R_1_peaks")

       
        if (len(R_0_peaks[0]) and len(R_1_peaks[0])) >0:
        
            lowest_peak0=1000
            for xx in R_0_peaks[0]:
                if R[0][xx]< lowest_peak0:
                    lowest_peak0=R[0][xx]
                    low_index0=xx
            lowest_peak1=1000
            for xx in R_1_peaks[0]:
               if R[1][xx]< lowest_peak1:
                   lowest_peak1=R[1][xx]
                   low_index1=xx
            
            vertical_dist=abs(lowest_peak1-lowest_peak0)
            horiz_dist= abs(self.lambda_[low_index1]-self.lambda_[low_index0])
            horiz_dist=horiz_dist*100000 #mult by 100000, the horizontal distance is so small it will be near zero otherwise
           
        else:
            vertical_dist=1
            horiz_dist=1
        ##fit each to a gaussian where it is centered
        parameters0, _ = curve_fit(Gauss, self.lambda_, R[0])
        parameters1, _ = curve_fit(Gauss, self.lambda_, R[1])

        fit_mean0 = parameters0[0]
        fit_sigma0 = parameters0[1]
        fit_mean1 = parameters1[0]
        fit_sigma1 = parameters1[1]

        fit_y0 = Gauss(self.lambda_, fit_mean0, fit_sigma0)
        mse0=( sum(R[0]-fit_y0)**2)/len(R[0])
        fit_y1 = Gauss(self.lambda_, fit_mean1, fit_sigma1)
        mse1=( sum(R[1]-fit_y1)**2)/len(R[1])
        print("sigmas")
        print("sigma0")
        print(fit_sigma0)
        print("sigma1")
        print(fit_sigma1)
        print("mean0")
        print(fit_mean0)
        print("mean1")
        print(fit_mean1)
        #output=sc.wasserstein_distance(R[0],R[1])
        print("vertical_dist")
        print(vertical_dist)
        print("horiz_distr")
        print(horiz_dist)
        print("mse0")
        print(mse0)
        print("mse1")
        print(mse1)
        number_of_peaks=len(R_0_peaks[0])+len(R_1_peaks[0])
        #output=-((2)*vertical_dist+(0.01)*horiz_dist -(0.002)*number_of_peaks)
        
        
       
        #output=-log((sc.wasserstein_distance(R[0],R[1])+(0.005)*vertical_dist))
        output=-log(vertical_dist)+log(mse0+mse1)-log(sc.wasserstein_distance(R[0],R[1]))
        # if horiz_dist>0:
        #     output=output-log(horiz_dist)
        # print("(parameter)*vertical_dist")
        # print(vertical_dist)
        # print("mse0")
        # print(mse0)
        # print("mse1")
        # print(mse1)
        # print("output")
        # print(output)
        # print("sc.wasserstein_distance(R[0],R[1])")
        # print(sc.wasserstein_distance(R[0],R[1]))
        # print("(parametter)*number_of_peaks")
        # print((0.0002)*number_of_peaks)
        ##output=(1/output)
        N=N-1 #remove when bug is gone
        material_dict_indices.pop(0)
        t.pop(0)
        
        return output
    
    
    # def objective_best(self, x):
        
        
        
    #     #This code for "p" polarization calculation 
    #     pol=x[0][1]
        
    #     N=x[0][0]
    #     N=N+1 #remove when bug is gone
        
    #     theta = np.array([-55,55])
    #     theta = theta*(pi/180)  # Angle of incidence [rad].
    #     #material_dict_indices=list(list(zip(*x))[0])
    #     material_dict_indices=x[1]
    #     material_dict_indices.insert(0,[(0,"Air","g")])#remove when bug is gone
    #     t=x[2]
    #     t.insert(0,1e-9)#remove when bug is gone
    #     omega=2*pi*c/self.lambda_
        
    #     R=np.empty((len(theta),len(self.lambda_)))
    #     for ii in range(0,len(theta)):
    #         for jj in range(0,len(self.lambda_)):
    #             x=Epsilon_Creator(dataframe=material_dict_df,lambda__=(self.lambda_[jj]),  N_=N,  material_dict_indices_=material_dict_indices)
    #             epsilon=x.create_epsilon()
    #             gamma=magnetophotonicCrystal(omega[jj], theta[ii], N, epsilon, t, pol)
    #             R[ii][jj]=(abs(gamma)**2 )
    #     energy=hbar*omega/eV
    #     R_0_peaks=ss.find_peaks(-1*R[0]) ##multiplied by -1 to find toughs

    #     R_1_peaks=ss.find_peaks(-1*R[1])
        
     

    #     def find_nearest(lst, value):
    #         idx=np.argmin(np.abs(np.array(lst)-value))
    #         return idx, lst[idx]


    #     if (len(R_0_peaks[0]) and len(R_1_peaks[0])) >0:

    #         lowest_peak0=1000
    #         for xx in R_0_peaks[0]:
    #             if R[0][xx]< lowest_peak0:
    #                 lowest_peak0=R[0][xx]
    #                 low_index0=xx
    #                 xlow0=energy[xx]
    #         lowest_peak1=lowest_peak0
    #         peak1_lowest=0
    #         for xx in R_1_peaks[0]:
    #             if R[1][xx]< lowest_peak1:
    #                 peak1_lowest=1
    #                 lowest_peak1=R[1][xx]
    #                 low_index1=xx
    #                 xlow1=energy[xx]
    #         if peak1_lowest==0:
    #             # _, low_index1 =find_nearest(R_1_peaks[0],low_index0)
    #             _, low_index1 =find_nearest(R[1],low_index0)
    #             lowest_peak1=R[1][low_index1]
    #             xlow1=energy[low_index1]
                
                
    #         elif peak1_lowest==1:
    #             #_, low_index0 =find_nearest(R_0_peaks[0],low_index1)
    #             _, low_index0 =find_nearest(R[0],low_index1)
    #             lowest_peak0=R[0][low_index0]
    #             xlow0=energy[low_index0]
    #         vertical_dist=abs(lowest_peak1-lowest_peak0)
      
            
    #     else:
    #         xlow0=0.12
    #         xlow1=0.12
    #         vertical_dist=1
        
            
       
    #     # Define the Gaussian function
    #     def Gauss0(x, A, B):
    #      	y = 1-A*np.exp(-1*B*x**2)
    #         return y
    #     def Gauss1(x, mean, sigma):
    #         return (1-((1/(sigma*math.sqrt(2*pi)))*np.exp(-0.5*np.square((x-mean)/sigma))))
    #     def guassianfunc(xVar, a, b, c,d):
    #         return a * 1-(np.exp(-(xVar - b) ** 2 / (2 * c ** 2))) + d 
    #     def guassianfunc_(xVar, a, b, c, d):
    #         return a * (np.exp(-(xVar - b) ** 2 / (2 * c ** 2))) + d 

    #     def Gauss(x, y):
    #         mean = np.sum(np.dot(x,y)) / np.sum(y)
    #         sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

            

    #         popt, _ = curve_fit(guassianfunc, x, y, p0=[max(y), mean, sigma,0],maxfev=7000)
          
    #         return popt
    #     Rp=R


    #     width=0.2# in ev change to m later
    #     #num_pts=4
    #     start_idx0=[i for i,v in enumerate(energy) if v<(xlow0-(width/2))] #change to be greater than when using lambda
    #     start_idx1=[i for i,v in enumerate(energy) if v<(xlow1-(width/2))] #change to be greater than when using lambda
    #     ####
    #     #start_idx0=low_index0 -num_pts
    #     #start_idx1=low_index1 -num_pts
    #     ###
    #     # print("xlow0")
    #     # print(xlow0)
    #     # print("energy")
    #     # print(energy)
    #     # print("start_idx0")
    #     # print(start_idx0)
    #     #########
    #     start_idx0=start_idx0[0]
    #     start_idx1=start_idx1[0]
    #     ###########
    #     #plt.plot(idx_start1, ylist[1], marker="x", markeredgecolor="red") 
    #     end_idx0 =[i for i,v in enumerate(energy) if v>(xlow0+(width/2))]#change to less than with lambda
    #     end_idx1 =[i for i,v in enumerate(energy) if v>(xlow1+(width/2))]#change to less than with lambda

    #     #end_idx0 =low_index0+num_pts
    #     #end_idx1= low_index1+num_pts

        
    #     ####
    #     end_idx0=end_idx0[len(end_idx0)-1]
    #     end_idx1=end_idx1[len(end_idx1)-1]
    #     ######
    #     print("start_idx0")
    #     print(start_idx0)
    #     print("end_idx0")
    #     print(end_idx0)

    #     print("start_idx1")
    #     print(start_idx1)
    #     print("end_idx1")
    #     print(end_idx1)
    #     #plt.plot(idx_end1, ylist[1], marker="x", markeredgecolor="red")
    #     energy_reduced0=energy[end_idx0:start_idx0] #switch these when using lambda
    #     energy_reduced1=energy[end_idx1:start_idx1] #switch these when using lambda
    #     R0_reduced=R[0][end_idx0:start_idx0]
    #     R1_reduced=R[1][end_idx1:start_idx1]
    #     #parameters0, _ = curve_fit(Gauss, energy_reduced, R0_reduced)
    #     #parameters1, _ = curve_fit(Gauss, energy_reduced, R1_reduced, p0=[max(R1_reduced), mean, sigma])

    #     #print(parameter1)
    #     #fit_mean0 = parameters0[0]
    #     #fit_sigma0 = parameters0[1]
    #     #fit_mean1 = parameters1[0]
    #     #fit_sigma1 = parameters1[1]
    #     print(R1_reduced)
    #     print("R1_reduced")
    #     if peak1_lowest==0:
    #         popt0=Gauss(np.array(energy_reduced0),np.array(R0_reduced))
    #         #popt1=Gauss(np.array(energy_reduced1),np.array(R1_reduced))
            
    #         shift0=1-R[0][end_idx0+5]
    #         #shift1=1-R[1][end_idx1+5]
           
    #         fit_y0=guassianfunc(energy,popt0[0]-shift0,xlow0,popt0[2],popt0[3]+shift0)
    #         fit_y0=fit_y0-min(fit_y0)
    #         #fit_y1=guassianfunc(energy,popt1[0]-shift1,popt1[1],popt1[2],popt1[3]+shift1)
    #         # fit_y1 =np.ones((1,len(energy)))
    #         fit_y1=R[1]/R[1]
    #         mse_low=( sum(R[0]-fit_y0)**2)/len(R[0])
            
    #         mse_high=( sum(R[1]-fit_y1)**2)/len(R[1])
    #         # print("fit_y1")
    #         # print(fit_y1)
    #     elif peak1_lowest==1:
    #         popt1=Gauss(np.array(energy_reduced0),np.array(R0_reduced))
    #         #popt1=Gauss(np.array(energy_reduced1),np.array(R1_reduced))
            
    #         shift1=1-R[1][end_idx1+5]
    #         #shift1=1-R[1][end_idx1+5]
           
    #         #fit_y0=guassianfunc(energy,popt0[0]-shift0,popt0[1],popt0[2],popt0[3]+shift0)
    #         fit_y1=guassianfunc(energy,popt1[0]-shift1,xlow1,popt1[2],popt1[3]+shift1)
    #         fit_y1=fit_y1-min(fit_y1)
    #         #fit_y0 =np.ones((1,len(energy)))
    #         fit_y0=R[0]/R[0]
    #         # print("fit_y0")
    #         # print(fit_y0)
    #         mse_high=( sum(R[0]-fit_y0)**2)/len(R[0])
            
    #         mse_low=( sum(R[1]-fit_y1)**2)/len(R[1])
       
        
    #     print("20*vertical_dist")
    #     print(30*vertical_dist)
    #     print("0.2*(mse_low)")
    #     print(0.2*(mse_low))
    #     print("0.2*(mse_high)")
    #     print(0.2*(mse_high))
    #     # print("mse1")
    #     # print(mse1)
    #     print("----------------")
      
    #     # print("----------------")
    #     # print("abs(popt1[0])*8*10**4")
    #     # print(abs(popt1[0])*8*10**4)
    #     # print("abs(popt0[0])*8*10**4")
    #     # print(abs(popt0[0])*8*10**4)
    #     # print("----------------")
    #     # print("abs(popt1[3]-popt0[3])*8*10**4")
    #     # print(abs(popt1[3]-popt0[3])*8*10**4)
        
    #     #print('0.001(mse0+mse1)')
    #     #print(0.1*(mse0+mse1))
    #     print("100*sc.wasserstein_distance(R[0],R[1])")
    #     print(100*sc.wasserstein_distance(R[0],R[1]))
        
    #     # print('abs(popt1[3])')
    #     # print(abs(popt1[3])*10**-4)
     
    #     #print("abs(parameters0[2])*10**4")
    #     #print(abs(parameters0[2])*10**4)
        

        
     
    #     #output=-log(20*vertical_dist)+log(0.4*(mse))-log(45*sc.wasserstein_distance(R0_reduced,R1_reduced))  #+log(abs(popt1[3])*10**-4)-log(abs(popt0[3])*10**-4)
    #     #output=-log(30*vertical_dist)+log(0.2*mse_low)+log(0.2*mse_high)-log(25*sc.wasserstein_distance(R0_reduced,R1_reduced))  #+log(abs(popt1[3])*10**-4)-log(abs(popt0[3])*10**-4)
    #     output=-log(30*vertical_dist)
    #     N=N-1 #remove when bug is gone
    #     material_dict_indices.pop(0)
    #     t.pop(0)
        
    #     return output
    
    
    def objective_one_with_extras(self, x, gen):
       
        
        output=0
        #This code for "p" polarization calculation 
        pol=x[0][1]
        
        N=x[0][0]
        #N=N+1 #remove when bug is gone
        
        theta = np.array([-55,55])
        theta = theta*(pi/180)  # Angle of incidence [rad].
        #material_dict_indices=list(list(zip(*x))[0])
        material_dict_indices=x[1]
        #material_dict_indices.insert(0,[(0,"Air","g")])#remove when bug is gone
        t=x[2]
        #t.insert(0,1e-9)#remove when bug is gone
        omega=2*pi*c/self.lambda_
        
        R=np.empty((len(theta),len(self.lambda_)))
        T=np.empty((len(theta),len(self.lambda_)))
        for ii in range(0,len(theta)):
            for jj in range(0,len(self.lambda_)):
                x=Epsilon_Creator(dataframe=material_dict_df,lambda__=(self.lambda_[jj]),  N_=N,  material_dict_indices_=material_dict_indices)
                epsilon=x.create_epsilon()
                R[ii][jj], T[ii][jj]=magnetophotonicCrystal(self.lambda_[jj], theta[ii], N, epsilon, t, pol)
                #gamma=magnetophotonicCrystal(omega[jj], theta[ii], N, epsilon, t, pol)
                #R[ii][jj]=(abs(gamma)**2 )
        energy=hbar*omega/eV
        R_0_peaks=ss.find_peaks(-1*R[0]) ##multiplied by -1 to find toughs

        R_1_peaks=ss.find_peaks(-1*R[1])
        print("R_1_peaks",R_1_peaks)

        def find_nearest(lst, value):
            idx=np.argmin(np.abs(np.array(lst)-value))
            return idx, lst[idx]


        if (len(R_0_peaks[0]) and len(R_1_peaks[0])) >0:

            lowest_peak0=1000
            for xx in R_0_peaks[0]:
                if R[0][xx]< lowest_peak0:
                    lowest_peak0=R[0][xx]
                    low_index0=xx
                    xlow0=energy[xx]
            lowest_peak1=lowest_peak0
            peak1_lowest=0
            for xx in R_1_peaks[0]:
                if R[1][xx]< lowest_peak1:
                    peak1_lowest=1
                    lowest_peak1=R[1][xx]
                    low_index1=xx
                    xlow1=energy[xx]
            if peak1_lowest==0:
                _, low_index11 =find_nearest(R_1_peaks[0],low_index0)## This is the nearest peak to the lowest peak, good for horizontal calculation
                lowest_peak1=R[1][low_index0] #R with same R index on opposite distribution of lowest peak
                #lowest_peak1=R[1][low_index1]
                xlow1=energy[low_index0]
                horiz_dist=abs(energy[low_index11]-energy[low_index0])
                
            elif peak1_lowest==1:
                _, low_index00 =find_nearest(R_0_peaks[0],low_index1)## This is the nearest peak to the lowest peak, good for horizontal calculation
               
                lowest_peak0=R[0][low_index1] #R with same R index on opposite distribution of lowest peak
                #lowest_peak0=R[0][low_index0]
                xlow0=energy[low_index1]
                horiz_dist=abs(energy[low_index1]-energy[low_index00])
                
            vertical_dist=abs(lowest_peak1-lowest_peak0)
       
            
        else:
           
            xlow0=0.12
            xlow1=0.12
            vertical_dist=0.0000001
            output=1000
            mse_low=1000
            mse_high=1000
            
       ################################################################################################
        # Define the Gaussian function
        def Gauss0(x, A, B):
            y = 1-A*np.exp(-1*B*x**2)
            return y
        def Gauss1(x, mean, sigma):
            return (1-((1/(sigma*math.sqrt(2*pi)))*np.exp(-0.5*np.square((x-mean)/sigma))))
        def guassianfunc(xVar, a, b, c,d):
            return a * 1-(np.exp(-(xVar - b) ** 2 / (2 * c ** 2))) + d 
        def guassianfunc_(xVar, a, b, c, d):
            return a * (np.exp(-(xVar - b) ** 2 / (2 * c ** 2))) + d 

        def Gauss(x, y):
            mean = np.sum(np.dot(x,y)) / np.sum(y)
            sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

            

            popt, _ = curve_fit(guassianfunc, x, y, p0=[max(y), mean, sigma,0],maxfev=7000)
           
            return popt
        Rp=R


        width=0.015# in ev change to m later
        #num_pts=4
        start_idx0=[i for i,v in enumerate(energy) if v<(xlow0-(width/2))] #change to be greater than when using lambda
        start_idx1=[i for i,v in enumerate(energy) if v<(xlow1-(width/2))] #change to be greater than when using lambda
        ####
        #start_idx0=low_index0 -num_pts
        #start_idx1=low_index1 -num_pts
        ###
        # print("xlow0")
        # print(xlow0)
        # print("energy")
        # print(energy)
        # print("start_idx0")
        # print(start_idx0)
        #########
        start_idx0=start_idx0[0]
        start_idx1=start_idx1[0]
        ###########
        #plt.plot(idx_start1, ylist[1], marker="x", markeredgecolor="red") 
        end_idx0 =[i for i,v in enumerate(energy) if v>(xlow0+(width/2))]#change to less than with lambda
        end_idx1 =[i for i,v in enumerate(energy) if v>(xlow1+(width/2))]#change to less than with lambda

        #end_idx0 =low_index0+num_pts
        #end_idx1= low_index1+num_pts
        # print("end_idx0")
        # print(end_idx0)
        # print("xlow0")
        # print(xlow0)
        ####
        print("end_idx0")
        print(end_idx0)
        end_idx0=end_idx0[len(end_idx0)-1]
        end_idx1=end_idx1[len(end_idx1)-1]
        ######
        # print("start_idx0")
        # print(start_idx0)
        # print("end_idx0")
        # print(end_idx0)

        # print("start_idx1")
        # print(start_idx1)
        # print("end_idx1")
        # print(end_idx1)
        #plt.plot(idx_end1, ylist[1], marker="x", markeredgecolor="red")
        energy_reduced0=energy[end_idx0:start_idx0] #switch these when using lambda
        energy_reduced1=energy[end_idx1:start_idx1] #switch these when using lambda
        R0_reduced=R[0][end_idx0:start_idx0]
        R1_reduced=R[1][end_idx1:start_idx1]
        #parameters0, _ = curve_fit(Gauss, energy_reduced, R0_reduced)
        #parameters1, _ = curve_fit(Gauss, energy_reduced, R1_reduced, p0=[max(R1_reduced), mean, sigma])

        #print(parameter1)
        #fit_mean0 = parameters0[0]
        #fit_sigma0 = parameters0[1]
        #fit_mean1 = parameters1[0]
        #fit_sigma1 = parameters1[1]
        # print(R1_reduced)
        # print("R1_reduced")
        if output!=1000:
            if peak1_lowest==0:
              
                popt0=Gauss(np.array(energy_reduced0),np.array(R0_reduced))
                #popt1=Gauss(np.array(energy_reduced1),np.array(R1_reduced))
                
                shift0=1-R[0][end_idx0+5]
                #shift1=1-R[1][end_idx1+5]
               
                fit_y0=guassianfunc(energy,popt0[0]-shift0,xlow0,popt0[2],popt0[3]+shift0)
                fit_y0=fit_y0-min(fit_y0)
                #fit_y1=guassianfunc(energy,popt1[0]-shift1,popt1[1],popt1[2],popt1[3]+shift1)
                # fit_y1 =np.ones((1,len(energy)))
                fit_y1=R[1]/R[1]
                
                
                #mse_low=( sum(R[0]-fit_y0)**2)/len(R[0])
                mse_low=mean_squared_error(R[0],fit_y0)
                
                #mse_high=( sum(R[1]-fit_y1)**2)/len(R[1])
                mse_high=mean_squared_error(R[1],fit_y1)
                # print("fit_y1")
                # print(fit_y1)
            elif peak1_lowest==1:
                
                popt1=Gauss(np.array(energy_reduced0),np.array(R0_reduced))
                #popt1=Gauss(np.array(energy_reduced1),np.array(R1_reduced))
                
                shift1=1-R[1][end_idx1+5]
                #shift1=1-R[1][end_idx1+5]
               
                #fit_y0=guassianfunc(energy,popt0[0]-shift0,popt0[1],popt0[2],popt0[3]+shift0)
                fit_y1=guassianfunc(energy,popt1[0]-shift1,xlow1,popt1[2],popt1[3]+shift1)
                fit_y1=fit_y1-min(fit_y1)
                #fit_y0 =np.ones((1,len(energy)))
                fit_y0=R[0]/R[0]
                # print("fit_y0")
                # print(fit_y0)
                
                
                #mse_high=( sum(R[0]-fit_y0)**2)/len(R[0])
                mse_high=mean_squared_error(R[0],fit_y0)
                #mse_low=( sum(R[1]-fit_y1)**2)/len(R[1])
                mse_low=mean_squared_error(R[1],fit_y1)
        
            print("30*vertical_dist")
            print(30*vertical_dist)
        #############################################################
            # print("0.1*(mse_low)")
            # print(0.1*(mse_low))
            # print("0.1*(mse_high)")
            # print(0.1*(mse_high))
            # print("mse1")
            # print(mse1)
            # print("----------------")
            # print("horiz_dist")
            # print(horiz_dist)
            # print("----------------")
            # print("abs(popt1[0])*8*10**4")
            # print(abs(popt1[0])*8*10**4)
            # print("abs(popt0[0])*8*10**4")
            # print(abs(popt0[0])*8*10**4)
            # print("----------------")
            # print("abs(popt1[3]-popt0[3])*8*10**4")
            # print(abs(popt1[3]-popt0[3])*8*10**4)
            
            #print('0.001(mse0+mse1)')
            #print(0.1*(mse0+mse1))
            # print("25*sc.wasserstein_distance(R[0],R[1])")
            # print(25*sc.wasserstein_distance(R[0],R[1]))
            
            # print('abs(popt1[3])')
            # print(abs(popt1[3])*10**-4)
         
            #print("abs(parameters0[2])*10**4")
            #print(abs(parameters0[2])*10**4)
            
    
            
         
            #output=-log(20*vertical_dist)+log(0.4*(mse))-log(45*sc.wasserstein_distance(R0_reduced,R1_reduced))  #+log(abs(popt1[3])*10**-4)-log(abs(popt0[3])*10**-4)
            # if gen<5:
            #     output=-log(10000000000000*vertical_dist)
            # else:
            #output=-log(100000000*vertical_dist)+log(0.0000001*mse_low)+log(0.0000001*mse_high)-log(0.1*sc.wasserstein_distance(R0_reduced,R1_reduced))  #+log(abs(popt1[3])*10**-4)-log(abs(popt0[3])*10**-4)
        #output=-log(100000000*vertical_dist)
        
        output=-log(1000000*vertical_dist)+log(0.0000001*mse_low)+log(0.0001*mse_high)#-log(0.1*sc.wasserstein_distance(R0_reduced,R1_reduced))
        mse_low=None
        mse_high=None
        R0_reduced=[0, 0, 0, 0 ]
        R1_reduced=[0, 0, 0, 0 ]
        # N=N-1 #remove when bug is gone
        # material_dict_indices.pop(0)
        # t.pop(0)
        
        return output,vertical_dist, mse_low, mse_high, sc.wasserstein_distance(R0_reduced,R1_reduced)


    
    
    def objective(self, x):
        
        output=0
        #This code for "p" polarization calculation 
        pol=x[0][1]
        
        N=x[0][0]
        #N=N+1 #remove when bug is gone
        
        theta = np.array([-55,55])
        theta = theta*(pi/180)  # Angle of incidence [rad].
        #material_dict_indices=list(list(zip(*x))[0])
        material_dict_indices=x[1]
        #material_dict_indices.insert(0,[(0,"Air","g")])#remove when bug is gone
        t=x[2]
        #t.insert(0,1e-9)#remove when bug is gone
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
                #gamma=magnetophotonicCrystal(omega[jj], theta[ii], N, epsilon, t, pol)
                #R[ii][jj]=(abs(gamma)**2 )
                
        Ap=1-(Rp+Tp)
        As=1-(Rs+Ts)
        energy=hbar*omega/eV
        
        
        vertical_dist=0.0000001
        output=1000
        mse_low=1000
        mse_high=1000
            
        # minimum1= min(Rp[1])
        # idx1=np.where(Rp[1]==minimum1)
        # idx1=idx1[0]
        # high1=Rp[0][idx1]
        # vertical_contrast1=high1-minimum1
        # db_value1=10*log((high1/minimum1))

        # minimum0= min(Rp[0])
        # idx0=np.where(Rp[0]==minimum0)
        # idx0=idx0[0]
        # high0=Rp[1][idx0]
        # vertical_contrast0=high0-minimum0
        # db_value0=10*log((high0/minimum0))
        contrast =abs(Ap[0]-Ap[1])
        maxcontrast =max(contrast)
        idx=np.where(contrast==maxcontrast)
        
        if Ap[0][idx]> Ap[1][idx]:
            FOM = (Ap[0][idx]+As[0][idx])/(Ap[1][idx]+As[1][idx])
        else: 
            FOM = (Ap[1][idx]+As[1][idx])/(Ap[0][idx]+As[0][idx])
        # if vertical_contrast1> vertical_contrast0:
        #     vertical_dist=vertical_contrast1
        #     FOM = (Rp[0][idx1]+Rs[0][idx1])/(Rp[1][idx1]+Rs[1][idx1])
           
        
        # else:
        #     vertical_dist=vertical_contrast0
        #     FOM = (Rp[1][idx0]+Rs[1][idx0])/(Rp[0][idx0]+Rs[0][idx0])
            
            
            
        # if vertical_contrast1> vertical_contrast0:
        #     vertical_dist=vertical_contrast1
        #     FOM = (Ap[1][idx0]+As[1][idx0])/(Ap[0][idx0]+As[0][idx0])
            
           
        
        # else:
        #     vertical_dist=vertical_contrast0
        #     FOM = (Ap[0][idx1]+As[0][idx1])/(Ap[1][idx1]+As[1][idx1])
           
           
           
        print("Vertical Dist")
        print(maxcontrast)
        
        
        #############################################################
            # print("0.1*(mse_low)")
            # print(0.1*(mse_low))
            # print("0.1*(mse_high)")
            # print(0.1*(mse_high))
            # print("mse1")
            # print(mse1)
            # print("----------------")
            # print("horiz_dist")
            # print(horiz_dist)
            # print("----------------")
            # print("abs(popt1[0])*8*10**4")
            # print(abs(popt1[0])*8*10**4)
            # print("abs(popt0[0])*8*10**4")
            # print(abs(popt0[0])*8*10**4)
            # print("----------------")
            # print("abs(popt1[3]-popt0[3])*8*10**4")
            # print(abs(popt1[3]-popt0[3])*8*10**4)
            
            #print('0.001(mse0+mse1)')
            #print(0.1*(mse0+mse1))
            # print("25*sc.wasserstein_distance(R[0],R[1])")
            # print(25*sc.wasserstein_distance(R[0],R[1]))
            
            # print('abs(popt1[3])')
            # print(abs(popt1[3])*10**-4)
         
            #print("abs(parameters0[2])*10**4")
            #print(abs(parameters0[2])*10**4)
            
    
            
         
            #output=-log(20*vertical_dist)+log(0.4*(mse))-log(45*sc.wasserstein_distance(R0_reduced,R1_reduced))  #+log(abs(popt1[3])*10**-4)-log(abs(popt0[3])*10**-4)
            # if gen<5:
            #     output=-log(10000000000000*vertical_dist)
            # else:
            #output=-log(100000000*vertical_dist)+log(0.0000001*mse_low)+log(0.0000001*mse_high)-log(0.1*sc.wasserstein_distance(R0_reduced,R1_reduced))  #+log(abs(popt1[3])*10**-4)-log(abs(popt0[3])*10**-4)
        # output=-log(vertical_dist)
        output =(FOM)
        # output=maxcontrast 
        mse_low=None
        mse_high=None
        R0_reduced=[0, 0, 0, 0 ]
        R1_reduced=[0, 0, 0, 0 ]
        # N=N-1 #remove when bug is gone
        # material_dict_indices.pop(0)
        # t.pop(0)
        
        return output,maxcontrast, mse_low, mse_high, sc.wasserstein_distance(R0_reduced,R1_reduced), x


    
    
    
    
   
    #tournament selection
    def selection(self, pop, scores, k=3):
      # print("inside selection")
      #first random selection
      selection_ix=randint(len(pop))
      for ix in randint(0,len(pop), k-1):
        if scores[ix] > scores[selection_ix]:
          selection_ix = ix
      return pop[selection_ix]
  

    # crossover two parents to create two children
    def crossover(self,p1, p2, r_cross):
        # print("inside crossover")
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        # print("P1")
        # print(p1[1])
        
        if len(p1[1])>3:
            if rand() < r_cross:
                list11=p1[1] #materials
                list21=p1[2] #thicknesses
                #material_thick_list1=zip(list11,list21) #zips the materials with their thicknesses for p1
                list12=p2[1] #materials
                list22=p2[2] #thicknesses
                #material_thick_list2=zip(list12,list22) #zips the materials with their thicknesses for p2
                # select crossover point that is not on the end of the string
                pt = randint(1, len(list11)-2) #subtract 2 so that the individuals will not be split at the ends
                # perform crossover
                c1 = [p1[0]]+[list11[:pt] + list12[pt:]]+[list21[:pt]+list22[pt:]]
                c2 = [p2[0]]+[list12[:pt] + list11[pt:]]+[list22[:pt]+list21[pt:]]
                
        
        else: 
            c1, c2 = p1, p2
        return [c1, c2]
     
    # mutation operator
    def mutation(self,attributes, r_mut):
        # print("inside mutation")
        # print("length of attributes")
        # print(len(attributes))
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
                        # list_=[]
                        unit_material_pair=[]
                        #attributes[i][ii]=random.choice(range(0,self.materials)) 
                        #attributes[i][ii]=random.choice(self.materials)
                        unit_cell=random.choice(self.unit_cell)
                        for jj in range(0,len(unit_cell)):
                            
                            
                            material=(random.choice(list(self.material_dict_[unit_cell[jj]].keys())) )
                            if rand() > self.g_flip:
                                g="g"
                            else:
                                g="-g"
                            unit_material_pair.append((unit_cell[jj],material, g))
                        attributes[i][ii]=unit_material_pair
                        # attributes[i][ii]=list_
                        # print("attributes[i][ii]")
                        # print(attributes[i][ii])
            elif i==2:
                for kk in range(0, len(attributes[i])-1):
                    if rand() < r_mut:
                            print('________THICKNESS MUTATION______________')
                            attributes[i][ii]=random.choice(self.thick)
        return attributes
                            
 ###########################################################################################                           
                            
    def genetic_algorithm(self, n_bits, n_iter, n_pop, r_cross, r_mut):
        # import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # current_working_directory = os.getcwd()
        # print("this path",current_working_directory)
        # path =current_working_directory
        # os.chdir(path)

        # with open('Iterations/'+str(timestr)+'.txt', 'w') as f:
        #     f.write("Genetic Algorithm: ")
            #f.write("output=-log(15*vertical_dist)+log(mse0+mse1)-log(sc.wasserstein_distance(R[0],R[1]))-log(abs(popt1[0])*8*10**4)+log(abs(popt0[0])*8*10**4)")
            
        # starting population of random attributes
        print("inside genetic algorithm")
        pop=self.population(n_pop)
        #print('pop')
        #print(pop)
        # keep track of best solution
        best_eval,L0,L1,L2,L3,L4 =  self.objective(pop[0]) #objective needs to take in inputs of the format we have
        best = pop[0]
        print("best")
        print(best)
        print("best_eval")
        print(best_eval)
        # enumerate generations
        counter_iter=0
        score_tracker=[]
        best_vert_dist =0
        # for gen in trange(n_iter):
        
        while best_vert_dist<=0.890:
        
    
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
                    # LL,L0,L1,L2,L3 =outputs[i]
                    best_vert_dist = outputs[i][1]
                    best_eval=copy.deepcopy(outputs[i][0])
                    best = outputs[i][5]
                    print(best,best_vert_dist)
                    print(" new best f(%s) = %.3f" % ( pop[i], outputs[i][0]))
                    with open('Iterations/'+str(timestr)+'.txt', 'a') as f:
                        f.write("\n")
                        f.write(str(best))
                        f.write("\n")
                        f.write("\n")
                        f.write('best_eval='+str(best_eval)+","+'vertical_dist='+str(L0)+","+ 'mse_low='+str(L1)+','+ 'mse_high='+str(L2)+','+ ' sc.wasserstein_distance(R0_reduced,R1_reduced))='+str(L3))
                        f.write("\n")
                        
            # select parents
            selected = [self.selection(pop, scores) for _ in range(n_pop)]
            # create the next generation
            children = list()
            for i in range(0, n_pop, 2):
                # get selected parents in pairs
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

            
# x=individual_maker(5,[2],['p','s'],[1,2,3,4])
# pop_n=5
# pop=x.population(pop_n)
# print("pop[0]")
# print(pop[0])

              
   

            
#TO DO:
#sometimes theres a problem with 2 layers-- unresolved?
#add more materials
#unit cell option
#problem with material 4 for some reason
#  ---run with x=Genetic_algo(4,[14],['p'],np.linspace(60,200,200)*(10**-9))
#keep the layer number the same in each individual in the population
#know how to construct an epsilon tensor when the material is anisotrophic

