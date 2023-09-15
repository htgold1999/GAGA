# IR-GAGA

## The General GAGA Algorithm
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

### The Materials 
All materials are included as .xls files in data. It is important that any additional materials be saved as .xls files, the material be added as a dielectric function in Epsilon_creator.py, and the material be added to the material dictionary function (material_dict()) Epsilon_creator.py.

--The numbers in the material_dict() function of Epsilon_creator.py are helpful for creating material pairings for the Genetic Algorithm to stick together.
--For example: [[1,3]] will randomly pair materials with the key '1' with '3', so dielectric materials can be paired with Weyl semimetaks.
This code uses a material library of SiO2, TiO2, MgO, and two Weyl Semimetals. There is additional data saved for Ag, HfO2, and Pt as well, but the material dictionary in Epsilon_creator.py would need to add those materials.
