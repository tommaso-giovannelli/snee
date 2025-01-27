					snee (Version 0.1, January 2025)
					================================
		  
						   Read me
						 ----------


snee is the name of the code associated with the manuscript entitled “Pareto sensitivity, 
most-changing sub-fronts, and knee solutions”.


snee is freely available for research, educational or commercial use, 
under a GNU lesser general public license.


This file describes the following topics:

1. System requirements
2. Code structure
3. How to reproduce the results in the manuscript
4. The snee team 


## 1. System requirements

The code is written in Python 3 (recommended version: Python 3.11.0). System requirements are available at 
https://www.python.org/downloads/release/python-3110

To run the code, install the required dependencies using the command: pip install -r requirements.txt.

## 2. Code structure

The code includes the following files:

  - **snee.py**: 
  
  Contains one class: 
	
 Snee (used to mplements the snee approach to compute most-changing sub-fronts and knee solutions for the problems defined in the functions.py file).

  - **functions.py**:	
  
  Contains several classes:
  
  SyntheticMultiobjProblem (used to define the problem for which we aim to determine most-changing sub-fronts and knee solutions)
	
ZLT1 (problem with 3 objectives)

GRV1 (problem with 3 objectives)

VFM1 (problem with 3 objectives)

ZLT1q (problem with q objectives)

GRV2 (problem with 2 objectives)

DAS1 (problem with 2 objectives) 

DO2DK (problem with 2 objectives) 

VFM1constr (problem with 3 objectives)

  - **driver.py**:
         
  To run the numerical experiments and obtain the figures in the manuscript, or to conduct custom experiments.

In the driver.py file, the dictionary exp_param_dict contains several parameters that can be used to define 
a numerical experiment. The values of such parameters can be set by modifying the values associated with the
corresponding keys, which are

   * name_prob_to_run_val (str):           A string representing the name of the problem for which we aim to determine most-changing sub-fronts and knee solutions	      
   * neighborhood_type (int):              Neighborhood type: 0 --> spherical; 1 --> ellipsoidal; 2 --> Cassini oval
   * compute_knee_solutions_flag (bool):   A flag to find knee solutions and the corresponding most-changing sub-front. If False, it only       
                                         computes the most-changing sub-front at the user weight vector in the functions.py file
   * algo_snee (str):                      The algorithm used to find knee solutions: "NM" --> The NM algorithm; "DIRECT" --> The DIRECT algorithm
   * plot_obj_funct_metric (bool):         A flag to plot the MCM and MCF over iterations
   * plot_neighborhood (bool):             A flag to plot the neighborhood in the parameter, objective, and decision spaces
   * iprint (int, optional):               Sets the verbosity level for printing information (higher values correspond to more detailed output) (default 1)

  - **requirements.txt**:

  Contains all the Python packages and their specific versions needed to run the code.

  - **README.md**:    
  
  The current file.


## 3. How to reproduce the results in the manuscript or conduct custom numerical experiments

To reproduce the results in the manuscript, comment out the lines related to the custom numerical experiments in the driver.py file, and then run it.

To conduct custom numerical experiments, comment out the lines related to the experiments from the paper in the driver.py file, and then run the file. 


## 4. The snee team 

   - Tommaso Giovannelli (University of Cincinnati)
   - Marcos Medeiros Raimundo (UNICAMP)
   - Luis Nunes Vicente (Lehigh University)



