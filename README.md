# IR-GAGA

## The General GAGA Algorithm
Coded by Hannah Gold
This is our Genetic Algorithm Gradient Ascent (GAGA) algorithm to optimize the structure of a 1D magnetophotonic crystal comprised of Weyl semimetals to maximize non-reciprocity.

### The Constraints
The constraints of our Genetic Algorithm in our run.py file are:<br>
**TOTAL ITERATIONS**   <br>
n_iter = 30  

**SIZE OF POPULATION** <br>
n_pop = 40   

**CROSSOVER RATE**  <br>
r_cross = 0.9

**MUTATION RATE** <br>
r_mut = 1/ 6

**G FLIP PROBABILITY** <br>
g_flip= 0.5

### How to Run
Run the run.py file and change the parameters if necessary. Additional materials can be added and paired with one another (see below described in materials). Iteration information is saved in the Iterations folder. The output will be first the material pairings (if specified, the materials will be paired and the number in front is not the layer number but the number associated with the material dictionary key), then the respective layer thicknesses. The order that the material and thicknesses are listed is from top of the structure to the bottom. There will always be one less thickness than material because the last layer is assumed to be semi-infinite.

### The Materials 
All materials are included as .xls files in data. It is important that any additional materials be saved as .xls files, the material be added as a dielectric function in Epsilon_creator.py, and the material be added to the material dictionary function (material_dict()) Epsilon_creator.py.

--The numbers in the material_dict() function of Epsilon_creator.py are helpful for creating material pairings for the Genetic Algorithm to stick together.
--For example: [[1,3]] will randomly pair materials with the key '1' with '3', so dielectric materials can be paired with Weyl semimetals.
This code uses a material library of SiO2, TiO2, MgO, and two Weyl semimetals. There is additional data saved for Ag, HfO2, and Pt as well, but the material dictionary in Epsilon_creator.py would need to add those materials.

## The Reflectance and Transmission Coefficients Calculation 
Coded by Simo Pajovic
This portion of the code is found in Magnetooptic2Magnetooptic.py and magnetophotonicCrystal.py and is a recursion which will calculate the reflectance and transmission coefficients. The result is completely analytical and can be generalized to all anisotropic media. The result is the same as the transfer matrix method. From these coefficients, the reflectance and transmittance can be found-- and since the absorptance = 1-(reflectance+transmittance), the absorptance is also easily calculated. This absorptance can be found for s- and p-polarizations respectively and is usd in our FOM. 
