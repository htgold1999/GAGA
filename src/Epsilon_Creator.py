# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:41:12 2022

@author: Hannah Gold
"""



#%%
import numpy as np 
import math
from math import *
import matplotlib.pyplot as plt
import cmath #
j=(cmath.sqrt(-1))



      # Physical constants.
hbar = 1.05457e-34
eV = 1.60218e-19
c = 2.99792e8
k_B = 1.38065e-23
sigma = 5.67037e-8
m_e = 9.10938e-31
epsilon_0 = 8.85419e-12


# Material properties.
class Epsilon_Creator():
    #lambda_ is in meters
    def __init__(self,dont_care=0, dataframe=None,lambda__=None,N_=None,material_dict_indices_=None):
       
        self.df=dataframe
        self.dont_care=dont_care #don't care about imaginary part
        self.lambda_=lambda__ #lambda
        #self.omega=omega #omega
        self.N=N_ #number of layers
        #self.material_dict=material_dict # A dictionary of materials and their dielectric fuctions
        self.material_dict_indices=material_dict_indices_ #list of keys corresponding to materials in the material dictionary
    def df_to_wavelength_list(self,df):
          wavelength_list= df["Wavelength"].astype(float).to_list()  #converts the dataframe column to a pandas series of floats then a list of floats     
         
          return wavelength_list
    def find_nearest(self,lst, value):
       
        idx=np.argmin(np.abs(np.array(lst)-value))
       
        
        return idx, lst[idx]
    
    def TiO2_eps(self,lambda_):
        # valid from 4.3e-7 m to 1.53e-6 m
        if  self.dont_care:
           
            material_df=self.df["TiO2"] 
            real_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_R"].astype(float).to_list())
            imag_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_I"].astype(float).to_list())
            eps_TiO2 = real_num 
           
        else:
           
            material_df=self.df["TiO2"] 
            real_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_R"].astype(float).to_list())
            imag_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_I"].astype(float).to_list())
            eps_TiO2 = real_num +j*imag_num
           
       
        return eps_TiO2
    def SiO2_eps(self,lambda_):
        
        
        
       
        if (lambda_*1000000)<7:
            eps_SiO2=1+((((0.6961663)*lambda_**2)/((lambda_**2)-(0.0684043**2))) + (((0.4079426)*lambda_**2)/((lambda_**2)-(0.1162414**2))) + (((0.8974794)*lambda_**2)/((lambda_**2)-(9.896161**2))))
            eps_SiO2=eps_SiO2+0.00014657*j
        elif  self.dont_care:
            # valid 2.1e-7 m to 67e-6 m
            material_df=self.df["SiO2"] 
            real_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_R"].astype(float).to_list())
            imag_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_I"].astype(float).to_list())
            eps_SiO2 = real_num 
           
            
        else:
           # valid 2.1e-7 m to 67e-6 m
            material_df=self.df["SiO2"] 
            real_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_R"].astype(float).to_list())
            imag_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_I"].astype(float).to_list())
            eps_SiO2 = real_num +j*imag_num
            
            
        return eps_SiO2
    def Si_eps(self,lambda_):
        
        
        if  self.dont_care:
            
            material_df=self.df["Si"] 
            real_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_R"].astype(float).to_list())
            # imag_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_I"].astype(float).to_list())
            eps_Si = real_num 
            
            
        else:
            
            material_df=self.df["Si"] 
            real_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_R"].astype(float).to_list())
            # imag_num=np.interp(lambda_,material_df["Wavelength"].astype(float).to_list(),material_df["Epsilon_I"].astype(float).to_list())
            eps_Si = real_num #+j*imag_num
            
           
        return eps_Si
    def HfO2_eps(self,lambda_):
        
        lambda_micrometer_to_m=lambda_*1000000
        if  False:
            eps_SiO2=1.46**2
            eps_SiO2=(1.7 +j*0.3)**2
            eps_SiO2=(1.7 )**2
        elif self.dont_care:
        
            material_df=self.df["HfO2"] # need to get refractive index given pandas dataframe
            wavelength_list=self.df_to_wavelength_list(material_df)
            idx,_=self.find_nearest(wavelength_list, lambda_micrometer_to_n)
            eps_HfO2=material_df["Epsilon_R"][idx] +j*material_df["Epsilon_Im"][idx] 
                
            
        return eps_HfO2
    
    
    def MgO_eps(self,lambda_):
        
        if (lambda_*1000000)< 10 and (lambda_*1000000)> 5.4:
            eps_MgO = 2.957019 + 0.0216485/(lambda_**2 - 0.0158650) - 0.0101373*lambda_**2 
            eps_MgO =eps_MgO 
        elif self.dont_care:
            # valid 3.6e-7 m to 5.4e-6 m
            material_df_0=self.df["MgO_n"] # need to get refractive index given pandas dataframe
            interp_n=np.interp(lambda_,material_df_0["Wavelength"].astype(float).to_list(),material_df_0["n"].astype(float).to_list())
            
            material_df_1=self.df["MgO_k"] # need to get refractive index given pandas dataframe
            interp_k=np.interp(lambda_,material_df_1["Wavelength"].astype(float).to_list(),material_df_1["k"].astype(float).to_list())
            eps_MgO = (interp_n+j*interp_k)**2
            eps_MgO = np.real(eps_MgO)
        else:
            # valid 3.6e-7 m to 5.4e-6 m
            material_df_0=self.df["MgO_n"] # need to get refractive index given pandas dataframe
            interp_n=np.interp(lambda_,material_df_0["Wavelength"].astype(float).to_list(),material_df_0["n"].astype(float).to_list())
            
            material_df_1=self.df["MgO_k"] # need to get refractive index given pandas dataframe
            interp_k=np.interp(lambda_,material_df_1["Wavelength"].astype(float).to_list(),material_df_1["k"].astype(float).to_list())
            
            eps_MgO = (interp_n+j*interp_k)**2
           
            #eps_MgO = np.real(eps_MgO)
            
        
        return eps_MgO
    def Pt_eps(self,lambda_):
        lambda_micrometer_to_m=lambda_*1000000
        if  0.00236 <= lambda_micrometer_to_m<=0.12157 or self.dont_care :
            omega_plasma=5.145*(eV/hbar) #Plasma Frequency of Pt [rad/sec]
            omega= (2*pi*c)/lambda_
            gamma_=(69.2*(10**-3))*(eV/hbar)  #drude damping
            eps_Pt= 1- ((omega_plasma**2)/((omega**2)+(j*omega*gamma_)))
        elif 0.248<= lambda_micrometer_to_m<= 12.4 :
            material_df=self.df["Pt"] # need to get refractive index given pandas dataframe
            wavelength_list=self.df_to_wavelength_list(material_df)
            idx,_=self.find_nearest(wavelength_list, lambda_micrometer_to_m)
            eps_Pt=material_df["Epsilon"][idx] 
        else:
            print("Other Pt_eps wavelengths not coded yet")
            eps_Pt=10000
        return eps_Pt
    
    def InAs_eps(self,lambda_):
        
       
        omega = 2*pi*c/lambda_
        B_field=0.12 # [T] Magnetic field
        eps_inf = 11.603 # [1] 
        omega_p = 0.347*eV # [J] plasma frequency
        gamma = 2.85e-3*eV  # [J] Drude damping constant for Fermi arc surface states
        n = (7.24*10**18)*10**6 # [m^-3] Carrier density
        m_eff = 0.083*m_e # [kg] Effective mass
        omega_c = B_field*eV/m_eff # [] #check units
        eps_t = eps_inf - ((omega_p**2)*(omega+j*gamma)/(omega*(((omega+j*gamma)**2) -omega_c**2)))

        g = (-j*(omega_p**2)*omega_c)/(omega*(((omega+j*gamma)**2) -omega_c**2))
        eps_l =eps_t     
        return eps_t,eps_l, g 
    
    # Look at this for different doping of InAs https://www.spiedigitallibrary.org/journals/journal-of-photonics-for-energy/volume-10/issue-02/025503/Design-of-an-indium-arsenide-cell-for-near-field-thermophotovoltaic/10.1117/1.JPE.10.025503.full?SSO=1
    
    
    
    def weyl_material_1_eps (self,lambda_):
      
        material_df=self.df["weyl_material_1"] # need to get refractive index given pandas dataframe
        real_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_R"])
        imag_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_I"])
        eps_t = real_num_t +j*imag_num_t
        
        real_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_R"])
        imag_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_I"])
        eps_l = real_num_l +j*imag_num_l
        
        g = np.interp(lambda_,material_df["Wavelength"],material_df["g"])
        
        
        
        # wavelength_list=self.df_to_wavelength_list(material_df)
        # idx,_=self.find_nearest(wavelength_list, lambda_)
        # eps_t=material_df["Epsilon_T_R"][idx] + j*material_df["Epsilon_T_I"][idx] 
        
        # eps_l=material_df["Epsilon_L_R"][idx] + j*material_df["Epsilon_L_I"][idx] 
        # g=material_df["g"][idx]
       
        return eps_t,eps_l, g 
    def weyl_material_1_b0_eps(self,lambda_):
        material_df=self.df["weyl_material_1_b0"] # need to get refractive index given pandas dataframe
        real_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_R"])
        imag_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_I"])
        eps_t = real_num_t +j*imag_num_t
        
        real_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_R"])
        imag_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_I"])
        eps_l = real_num_l +j*imag_num_l
        
        g = np.interp(lambda_,material_df["Wavelength"],material_df["g"])
        
        
        
        # wavelength_list=self.df_to_wavelength_list(material_df)
        # idx,_=self.find_nearest(wavelength_list, lambda_)
        # eps_t=material_df["Epsilon_T_R"][idx] + j*material_df["Epsilon_T_I"][idx] 
        
        # eps_l=material_df["Epsilon_L_R"][idx] + j*material_df["Epsilon_L_I"][idx] 
        # g=material_df["g"][idx]
       
        return eps_t,eps_l, g 
    def weyl_material_1_EF_01_eps (self,lambda_):
       
        material_df=self.df["weyl_material_1_EF=0.1"] # need to get refractive index given pandas dataframe
        real_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_R"])
        imag_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_I"])
        eps_t = real_num_t +j*imag_num_t
        
        real_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_R"])
        imag_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_I"])
        eps_l = real_num_l +j*imag_num_l
        
        g = np.interp(lambda_,material_df["Wavelength"],material_df["g"])
        
        
        
        # wavelength_list=self.df_to_wavelength_list(material_df)
        # idx,_=self.find_nearest(wavelength_list, lambda_)
        # eps_t=material_df["Epsilon_T_R"][idx] + j*material_df["Epsilon_T_I"][idx] 
        
        # eps_l=material_df["Epsilon_L_R"][idx] + j*material_df["Epsilon_L_I"][idx] 
        # g=material_df["g"][idx]
       
        return eps_t,eps_l, g 
    def weyl_material_1_EF_007_eps (self,lambda_):
        # lambda_micrometer_to_m=lambda_*1000000
        material_df=self.df["weyl_material_1_EF=0.07"] # need to get refractive index given pandas dataframe
        real_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_R"])
        imag_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_I"])
        eps_t = real_num_t +j*imag_num_t
        
        real_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_R"])
        imag_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_I"])
        eps_l = real_num_l +j*imag_num_l
        
        g = np.interp(lambda_,material_df["Wavelength"],material_df["g"])
        
        
        
        # wavelength_list=self.df_to_wavelength_list(material_df)
        # idx,_=self.find_nearest(wavelength_list, lambda_)
        # eps_t=material_df["Epsilon_T_R"][idx] + j*material_df["Epsilon_T_I"][idx] 
        
        # eps_l=material_df["Epsilon_L_R"][idx] + j*material_df["Epsilon_L_I"][idx] 
        # g=material_df["g"][idx]
       
        return eps_t,eps_l, g 
    def weyl_material_1_EF_02_eps (self,lambda_):
        # lambda_micrometer_to_m=lambda_*1000000
        material_df=self.df["weyl_material_1_EF=0.2"] # need to get refractive index given pandas dataframe
        real_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_R"])
        imag_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_I"])
        eps_t = real_num_t +j*imag_num_t
        
        real_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_R"])
        imag_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_I"])
        eps_l = real_num_l +j*imag_num_l
        
        g = np.interp(lambda_,material_df["Wavelength"],material_df["g"])
        
        
        
        # wavelength_list=self.df_to_wavelength_list(material_df)
        # idx,_=self.find_nearest(wavelength_list, lambda_)
        # eps_t=material_df["Epsilon_T_R"][idx] + j*material_df["Epsilon_T_I"][idx] 
        
        # eps_l=material_df["Epsilon_L_R"][idx] + j*material_df["Epsilon_L_I"][idx] 
        # g=material_df["g"][idx]
       
        return eps_t,eps_l, g
    def weyl_material_2_eps (self,lambda_):
        # lambda_micrometer_to_m=lambda_*1000000
        material_df=self.df["weyl_material_2"] # need to get refractive index given pandas dataframe
            
        real_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_R"])
        imag_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_I"])
        eps_t = real_num_t +j*imag_num_t
        
        real_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_R"])
        imag_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_I"])
        eps_l = real_num_l +j*imag_num_l
        
        g = np.interp(lambda_,material_df["Wavelength"],material_df["g"])
        
        # wavelength_list=self.df_to_wavelength_list(material_df)
        # idx,_=self.find_nearest(wavelength_list, lambda_)
        # eps_t=material_df["Epsilon_T_R"][idx] + j*material_df["Epsilon_T_I"][idx] 
        # # print("eps_t")
        # # print(eps_t)
        # eps_l=material_df["Epsilon_L_R"][idx] + j*material_df["Epsilon_L_I"][idx] 
        # g=material_df["g"][idx]
        
        return eps_t,eps_l, g # z and x are transverse and longitudinal components (respectively) of the dielectric tensor see source https://dspace.mit.edu/bitstream/handle/1721.1/132930/PhysRevB.102.165417.pdf?sequence=1
    def weyl_material_2_b0_eps(self,lambda_):
        material_df=self.df["weyl_material_2_b0"] # need to get refractive index given pandas dataframe
        real_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_R"])
        imag_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_I"])
        eps_t = real_num_t +j*imag_num_t
        
        real_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_R"])
        imag_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_I"])
        eps_l = real_num_l +j*imag_num_l
        
        g = np.interp(lambda_,material_df["Wavelength"],material_df["g"])
        
        
        
        # wavelength_list=self.df_to_wavelength_list(material_df)
        # idx,_=self.find_nearest(wavelength_list, lambda_)
        # eps_t=material_df["Epsilon_T_R"][idx] + j*material_df["Epsilon_T_I"][idx] 
        
        # eps_l=material_df["Epsilon_L_R"][idx] + j*material_df["Epsilon_L_I"][idx] 
        # g=material_df["g"][idx]
       
        return eps_t,eps_l, g 
    def weyl_material_2_EF_01_eps (self,lambda_):
        # lambda_micrometer_to_m=lambda_*1000000
        material_df=self.df["weyl_material_2_EF=0.1"] # need to get refractive index given pandas dataframe
        real_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_R"])
        imag_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_I"])
        eps_t = real_num_t +j*imag_num_t
        
        real_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_R"])
        imag_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_I"])
        eps_l = real_num_l +j*imag_num_l
        
        g = np.interp(lambda_,material_df["Wavelength"],material_df["g"])
        
        
        
        # wavelength_list=self.df_to_wavelength_list(material_df)
        # idx,_=self.find_nearest(wavelength_list, lambda_)
        # eps_t=material_df["Epsilon_T_R"][idx] + j*material_df["Epsilon_T_I"][idx] 
        
        # eps_l=material_df["Epsilon_L_R"][idx] + j*material_df["Epsilon_L_I"][idx] 
        # g=material_df["g"][idx]
       
        return eps_t,eps_l, g 
    def weyl_material_2_EF_007_eps (self,lambda_):
        # lambda_micrometer_to_m=lambda_*1000000
        material_df=self.df["weyl_material_2_EF=0.07"] # need to get refractive index given pandas dataframe
        real_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_R"])
        imag_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_I"])
        eps_t = real_num_t +j*imag_num_t
        
        real_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_R"])
        imag_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_I"])
        eps_l = real_num_l +j*imag_num_l
        
        g = np.interp(lambda_,material_df["Wavelength"],material_df["g"])
        
        
        
        # wavelength_list=self.df_to_wavelength_list(material_df)
        # idx,_=self.find_nearest(wavelength_list, lambda_)
        # eps_t=material_df["Epsilon_T_R"][idx] + j*material_df["Epsilon_T_I"][idx] 
        
        # eps_l=material_df["Epsilon_L_R"][idx] + j*material_df["Epsilon_L_I"][idx] 
        # g=material_df["g"][idx]
       
        return eps_t,eps_l, g 
    def weyl_material_2_EF_02_eps (self,lambda_):
        # lambda_micrometer_to_m=lambda_*1000000
        material_df=self.df["weyl_material_2_EF=0.2"] # need to get refractive index given pandas dataframe
        real_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_R"])
        imag_num_t=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_T_I"])
        eps_t = real_num_t +j*imag_num_t
        
        real_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_R"])
        imag_num_l=np.interp(lambda_,material_df["Wavelength"],material_df["Epsilon_L_I"])
        eps_l = real_num_l +j*imag_num_l
        
        g = np.interp(lambda_,material_df["Wavelength"],material_df["g"])
        
        
        
        # wavelength_list=self.df_to_wavelength_list(material_df)
        # idx,_=self.find_nearest(wavelength_list, lambda_)
        # eps_t=material_df["Epsilon_T_R"][idx] + j*material_df["Epsilon_T_I"][idx] 
        
        # eps_l=material_df["Epsilon_L_R"][idx] + j*material_df["Epsilon_L_I"][idx] 
        # g=material_df["g"][idx]
       
        return eps_t,eps_l, g 
    
    def Ag_eps(self,lambda_):
        lambda_micrometer_to_m=lambda_*1000000
        if  0.1879<= lambda_micrometer_to_m<=1.9370 or 2.480e-06<= lambda_micrometer_to_m<=248 or 0.270<= lambda_micrometer_to_m<=24.9 or self.dont_care:
            material_df=self.df["Ag"] # need to get refractive index given pandas dataframe
            wavelength_list=self.df_to_wavelength_list(material_df)
            idx,_=self.find_nearest(wavelength_list, lambda_micrometer_to_m)
            eps_Ag=material_df["Epsilon"][idx] 
        else:
            print("Other Ag_eps wavelengths not coded yet")
            eps_Ag=10000
        
        return eps_Ag
    def Au_eps(self,lambda_):
        lambda_micrometer_to_m=lambda_*1000000
        if  0.667<= lambda_micrometer_to_m<=286 or self.dont_care:
            material_df=self.df["Au"] # need to get refractive index given pandas dataframe
            wavelength_list=self.df_to_wavelength_list(material_df)
            idx,_=self.find_nearest(wavelength_list, lambda_micrometer_to_m)
            eps_Au=material_df["Epsilon"][idx] 
        else:
            print("Other Au_eps wavelengths not coded yet")
            eps_Au=10000
    def Air(self,lambda_):
        eps_Air=1
        
        return eps_Air
   
    def material_dict(self):
        material_dict = { 0:{"Air": {"function": self.Air, "isotropic":1}},1:{"TiO2": {"function": self.TiO2_eps, "isotropic":1}, "SiO2": {"function":self.SiO2_eps,"isotropic":1},'MgO': {"function":self.MgO_eps,"isotropic":1}}, 
                          2:{"Ag": {"function":self.Ag_eps,"isotropic":1}, "Pt": {"function":self.Pt_eps,"isotropic":1}},
                          3:{"weyl_material_1": {"function":self.weyl_material_1_eps,"isotropic":0},
                             "weyl_material_2": {"function":self.weyl_material_2_eps,"isotropic":0}},
                          4:{"Si": {"function":self.Si_eps,"isotropic":1}},5:{"InAs": {"function":self.InAs_eps,"isotropic":0}}}
        
    
       
        
        
        
        return material_dict
    def create_epsilon(self):
        material_dict=self.material_dict()
        epsilon=np.zeros((self.N,3),dtype=np.complex_)
      
        count=0
        flat_list = [item for sublist in self.material_dict_indices for item in sublist]
       
        for ii in flat_list: 
           
            if material_dict[ ii[0]][ ii[1]]["isotropic"]==1:
                g=0
                value=material_dict[ ii[0]][ ii[1]]["function"](self.lambda_)
                
                epsilon[count][0]= value
                
                epsilon[count][1]= value
                if ii[2]=='g':
                    epsilon[count][2]= g
                if ii[2]=='-g':
                    epsilon[count][2]= -g
             
                
            elif material_dict[ ii[0]][ ii[1]]["isotropic"]==0:
                 eps_T,eps_L, g=material_dict[ ii[0]][ ii[1]]["function"](self.lambda_)
               
                 epsilon[count][0]= eps_T
                
                 epsilon[count][1]= eps_L
                 
                 if ii[2]=='g':
                     epsilon[count][2]= g
                 if ii[2]=='-g':
                     epsilon[count][2]= -g
                
                 
            else:
                epsilon[count][0]= 0
                epsilon[count][1]= 0
                epsilon[count][2]= 0
            count+=1
        
        return np.round_(epsilon,decimals = 4)
    
    
    
    
