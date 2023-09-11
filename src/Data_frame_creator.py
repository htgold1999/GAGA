# # -*- coding: utf-8 -*-
# """
# Created on Mon Oct 24 09:19:43 2022

# @author: Hannah Gold
# """
#%% 
#from Epsilon_Creator import *
import os
import glob
import pandas as pd
import numpy as np

###NOTE Be sure to name all refractive index .xls files with their material name matching the one in material_dict (located in Epsilon_Creator.py )
## if the material is isotropic, the dataframe (which extracts from a csv) only needs two columns: wavelength (nm), and epsilon value
## if the material is anisotropic, the dataframe (which extracts from a csv) needs wavelength, epsilon transverse, epsilon, longitudinal, off diagonal (g) components

current_working_directory = os.getcwd()

#path =current_working_directory+'/src/Find_reflectance/Refractive_index_data/' 
path =current_working_directory+'/Find_reflectance/Refractive_index_data/' 
os.chdir(path)

def Data_frame_creator(path):
    #returns a dictionary of dataframes
    
    all_files = glob.glob(path + "/*.xls") #includes file directory with .csv file
    
    
    
    filenames=[os.path.splitext(os.path.basename(x))[0] for x in all_files]
    
    #li = []
    
    material_dict_df=dict()
    for filename_and_path, filename in zip(all_files, filenames):
        #print(filename)
        #read file here
        # if you decide to use pandas you might need to use the 'sep' paramaeter as well
        df = pd.read_excel(filename_and_path, index_col=None, header=0)
        material_dict_df[filename]=df
        #df= np.genfromtxt(filename)
    
        #li.append(df)
    
        # get it all together
        #frame = pd.concat(li, axis=0, ignore_index=True) ## uncomment this if you want the dataframes concatonated together
    return material_dict_df
    #a = pd.read_csv('C:/Users/Hannah Gold/Desktop/MIT/MIT2022_2023/Nonreciprocal_Magnetooptic/src/Find_reflectance/Refractive_index_data/Ag_RefractiveIndex.csv')
    
    #dictionary= { "TiO2": {"material": "TiO2_eps", "isotropic":1, "dataframe":a }, "SiO2": {"material":"SiO2_eps","isotropic":1, "dataframe":a}}




material_dict_df=Data_frame_creator(path)
os.chdir('../')

#print(material_dict_df["Ag_RefractiveIndex"]["Wavelength"].to_list()[0])
#print(material_dict_df["Ag_RefractiveIndex"]["Wavelength"].to_numpy()[0])


# print("test test")
# df=material_dict_df["SiO2"]
# print(df["Epsilon"].astype(float).to_list())