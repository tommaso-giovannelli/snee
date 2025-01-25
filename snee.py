import os
import numpy as np
import random
import itertools
from scipy.optimize import minimize
from scipy.optimize import direct, Bounds
from scipy.optimize import NonlinearConstraint
import torch

class Snee:
    """
    Class that implements the snee approach to compute most-changing sub-fronts and knee solutions for the problems defined in the functions.py file

    Attributes
        func (obj):                           Object used to define the optimization problem for which we want to determine most-changing sub-fronts and knee solutions
        num_pareto_points (int):              Number of weights in the discretization of the simplex set       
        neighborhood_type (int):              Neighborhood type: 0 --> spherical; 1 --> ellipsoidal; 2 --> Cassini oval
        compute_knee_solutions_flag (bool):   A flag to find knee solutions and the corresponding most-changing sub-front. If False, it only computes the most-changing sub-front at the user weight vector in the functions.py file
        algo_snee (str):                      The algorithm used to find knee solutions: "NM" --> The NM algorithm; "DIRECT" --> The DIRECT algorithm
        plot_obj_funct_metric (bool):         A flag to plot the MCM and MCF over iterations
        plot_neighborhood (bool):             A flag to plot the neighborhood in the parameter, objective, and decision spaces
        iprint (int, optional):               Sets the verbosity level for printing information (higher values correspond to more detailed output) (default 1) 
        seed (int, optional):                 The seed used for the experiments (default 0)
    """    
    
    
    def __init__(self, func, \
                 neighborhood_type, \
                 compute_knee_solutions_flag, \
                 algo_snee, \
                 plot_obj_funct_metric, \
                 plot_neighborhood, \
                 iprint = 1, \
                 seed = 0):


        self.func = func
        self.num_pareto_points = self.func.prob.num_pareto_points 
        self.neighborhood_type = neighborhood_type
        self.compute_knee_solutions_flag = compute_knee_solutions_flag
        self.algo_snee = algo_snee
        self.plot_obj_funct_metric = plot_obj_funct_metric
        self.plot_neighborhood = plot_neighborhood  
        self.iprint = iprint
        self.seed = seed        
        
        
    def set_seed(self, seed):
        """
        Sets the seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed) 
    
    
    def main_snee(self):
        """
        Returns:
            - f_value_list_dict:                 Dictionary of lists of values for each objective function for each weight vector in the simplex set
            - y_list:                            List of Pareto solutions for each weight vector in the simplex set
            - simplex_set_discr:                 Array where each column is a weight vector in the simplex set
            - f_value_dict_user_per_weight:      Objective function values associated with the user weight vector
            - y_user_per_weight:                 Pareto solution associated with the user weight vector
            - weight_user:                       User weight vector
            - f_value_neighborhood_list_dict:    Dictionary of lists that contain objective function values in the neighborhood in the objective space
            - y_neighborhood_list:               List that contains Pareto solutions in the decision space
            - simplex_set_discr_neighborhood:    Two-dimensional array where each column is a weight vector in the Pareto neighborhood in the parameter space
            - A:                                 Matrix used to define the neighborhood 
            - B:                                 Jacobian of the Pareto solution y wrt the weight vector
            - rhs_ellipsoid:                     RHS of the neighborhood
            - weight_iterates_list:              List that contains the weight vectors across the iterations of the optimization algotithm used to find knee solutions
            - f_value_iterates_list:             List that contains the objective function values across the iterations of the optimization algotithm used to find knee solutions
            - y_iterates_list:                   List that contains the values of the decision variables across the iterations of the optimization algotithm used to find knee solutions
            - obj_funct_values_list:             List of MCF values 
            - metric_values_list:                List of MCM values
            - value_test_obj_list
        """
        self.set_seed(self.seed) 
        
        f_value_list_dict = {}
        y_list = []
        
        ## Create an empty list for each objective
        for m in range(self.func.prob.num_obj):
            f_value_list_dict[m] = []

        ## Initialize the vector in the decision space used as the initial point when applying the weighted-sum method for a given weight vector
        y_init = np.random.uniform(0, 1, (self.func.prob.dim, 1))       
        # y_init = self.func.prob.y_init
        y_init = y_init.reshape(-1,1)
        ###############################

        ## Generate weight vectors equidistantly within the simplex set 
        simplex_set_discr = np.empty((self.func.prob.num_obj, 0))
        for vector in itertools.product(*(np.linspace(0, 1, self.num_pareto_points) for _ in range(self.func.prob.num_obj-1))):
          rnd = np.array(sorted([0] + list(vector) + [1]))
          lam = np.array([rnd[i+1]-rnd[i] for i in range(self.func.prob.num_obj)])
          lam = lam/lam.sum()
          simplex_set_discr = np.hstack((simplex_set_discr,lam.reshape(-1,1)))         
        # Remove repeated columns from simplex_set_discr
        # Transpose the array to work with rows (original columns)
        transposed_array = np.transpose(simplex_set_discr)     
        # Use np.unique to find unique rows
        unique_rows = np.unique(transposed_array, axis=0)
        # Transpose back to get the original shape
        simplex_set_discr = np.transpose(unique_rows)
        ###############################     
        
        ## Apply the weighted-sum method for each weight vector in the simplex set and store the objective function values and Pareto solutions
        for lam in simplex_set_discr.T:  
           out = self.weighted_sum_method(y_init, lam)
           # print("check_kkt_system: ",self.func.check_kkt_system(lam, out[1]).T)
           
           for m in range(self.func.prob.num_obj):
               f_value_list_dict[m].append(out[0][m])
               y_list.append(out[1])
        ###############################
            
        weight_user = self.func.prob.weight_user 
        
        if self.iprint >= 1:
            print('\nUser Weight: ',weight_user,'\n')
        
        ## Determine the knee solution by solving the optimization problem of the snee approach 
        # IT WILL MODIFY weight_user WITH THE OPTIMAL ONE
        weight_iterates_list = []
        f_value_iterates_list = []
        y_iterates_list = []
        obj_funct_values_list = []
        metric_values_list = []
        value_test_obj_list = []

        if self.compute_knee_solutions_flag: 
            weight_var_optimized, weight_iterates_list, obj_funct_values_list, metric_values_list, value_test_obj_list = self.compute_knee_solutions(y_init, weight_user[0], simplex_set_discr, f_value_list_dict)
            weight_user[0] = weight_var_optimized.reshape(-1,1)
            
            if self.iprint >= 1:
                print('\nOptimal weight vector (associated with the knee solution): ',weight_user,'\n')
            
            for weigth_iterate in weight_iterates_list:
                f_value_iterate, y_iterate = self.weighted_sum_method(y_init, weigth_iterate.reshape(-1,1))
                f_value_iterates_list.append(f_value_iterate)
                y_iterates_list.append(y_iterate)
        ###############################
    
        ## Determine the Pareto neighborhood associated with the knee solution in the parameter, objective, and decision spaces
        simplex_set_discr_neighborhood, f_value_neighborhood_list_dict, y_neighborhood_list, \
            f_value_dict_user_per_weight, y_user_per_weight, A, rhs_ellipsoid = self.compute_mostchanging_subfronts(weight_user,y_init,simplex_set_discr,iprint=True)
        ###############################

        metric = self.MCM(f_value_list_dict,f_value_neighborhood_list_dict)
        
        if self.iprint >= 1:
            print('\nMCM: {0}'.format(metric),'\n')
    
        ## Remove from simplex_set_discr all the columns (weight vectors) that are in simplex_set_discr_neighborhood for plotting purposes 
        if self.plot_neighborhood:
            # Find columns in array1 that are not in array2
            # Convert the columns of A and B to tuples for comparison
            tuples_1 = [tuple(row) for row in simplex_set_discr.T]
            tuples_2 = [tuple(row) for row in simplex_set_discr_neighborhood.T]           
            # Find the columns in A that are not in B
            simplex_set_discr = simplex_set_discr[:, [col not in tuples_2 for col in tuples_1]]
        ############################### 

        ## Compute the Jacobian of the Pareto solution y wrt the weight vector
        jacob_F_y = self.func.jacob_F_y(weight_user[0], y_user_per_weight[0])
        B = jacob_F_y
        ###############################
            
        ## Remove from f_value_list_dict[m] all the columns (objective function values) that are in f_value_neighborhood_list_dict[m] for plotting purposes
        if self.plot_neighborhood:
            # Find columns in X that are not in Y
            # Convert the columns of X and Y to tuples for comparison
            X = np.vstack(list(f_value_list_dict.values()))
            Y = np.vstack(list(f_value_neighborhood_list_dict.values()))
            tuples_1 = [tuple(row) for row in X.T]
            tuples_2 = [tuple(row) for row in Y.T]           
            # Find the columns in X that are not in Y
            X = X[:, [elem not in tuples_2 for elem in tuples_1]]
            for i in range(X.shape[0]):
                f_value_list_dict[i] = X[i]
        ###############################

        ## Remove from y_list all the Pareto solutions that are in y_neighborhood_list for plotting purposes
        if self.plot_neighborhood:                
            # Flatten the arrays for easy comparison
            y_ellipsoid_flat = [arr.flatten() for arr in y_neighborhood_list]
            y_list_flat = [arr.flatten() for arr in y_list]          
            # Filter out arrays in y_list that are in y_ellipsoid_flat
            filtered_y_list = [arr for arr in y_list_flat if not np.any(np.all(arr == y_ellipsoid_flat, axis=1))]
            # Reshape the filtered arrays back to their original shape
            y_list = [arr.reshape(y_neighborhood_list[0].shape) for arr in filtered_y_list]
        ###############################
        
        return f_value_list_dict, y_list, simplex_set_discr, \
            f_value_dict_user_per_weight, y_user_per_weight, weight_user,\
            f_value_neighborhood_list_dict, y_neighborhood_list, simplex_set_discr_neighborhood, \
            A, B, rhs_ellipsoid, \
            weight_iterates_list, f_value_iterates_list, y_iterates_list, obj_funct_values_list, metric_values_list, value_test_obj_list
  
    
    def weighted_sum_method(self, y_k, lam):
        """
        Returns the Pareto solution in the decision space obtained by applying the weighted sum method from y_k with weight vector given by lam.
        It also returns the list of objective function values evaluated at such a solution. 
        We use BFGS in the unconstrained case and SLSQP in the constrained case. 
        """
        ## Dictionary that contains the function values at the Pareto point for each objective
        f_value_dict = {} 
        
        if self.func.prob.constrained:
            #---------------------------------------------#
            #-------------- Constrained Case -------------#
            #---------------------------------------------#
            ineq_constraint = NonlinearConstraint(lambda y: self.func.prob.inequality_constraints(y).flatten(), -np.inf, 0, jac=lambda y: self.func.prob.inequality_constraints_jacob(y).T, hess=lambda y, lagr_multiplier: self.func.prob.inequality_constraints_hess_with_multipliers(y, lagr_multiplier))  
            constr_list = [ineq_constraint]
            if self.func.prob.num_eq_constr != 0:
                eq_constraint = NonlinearConstraint(lambda y: self.func.prob.equality_constraints(y).flatten(), 0, 0, jac=lambda y: self.func.prob.equality_constraints_jacob(y).T, hess=lambda y, lagr_multiplier: self.func.prob.equality_constraints_hess_with_multipliers(y, lagr_multiplier))
                constr_list = [ineq_constraint, eq_constraint]
            
            result = minimize(
                fun= lambda y: self.func.f_weighted(lam,y),
                x0=y_k.flatten(),
                method='SLSQP',
                jac= lambda y: self.func.grad_f_weighted_vars(lam,y),
                # hess= lambda y: self.func.hess_f_weighted_vars(lam,y),
                constraints=constr_list
            )
            
            y_k = result.x.reshape(-1,1)

        else:
            #---------------------------------------------#
            #------------- Unconstrained Case ------------#
            #---------------------------------------------#
            result = minimize(lambda y: self.func.f_weighted(lam,y.reshape(-1,1)).flatten(), x0=y_k.flatten(), method='BFGS', jac=lambda y: self.func.grad_f_weighted_vars(lam,y.reshape(-1,1)).flatten())
            y_k = result.x.reshape(-1,1)
           
        for m in range(self.func.prob.num_obj):
            f_value_dict[m] = self.func.prob.f_dict[m](y_k.reshape(-1,1))
           
        return f_value_dict, y_k


    def compute_mostchanging_subfronts(self,weight_user,y_init,simplex_set_discr,knee_solutions_flag=False,iprint=False):
        """
        Returns:
            - simplex_set_discr_neighborhood:       Two-dimensional array where each column is a weight vector in the Pareto neighborhood in the parameter space
            - f_value_neighborhood_list_dict:       Dictionary of lists that contain objective function values in the neighborhood in the objective space
            - y_neighborhood_list:                  List that contains Pareto solutions in the decision space
            - f_value_dict_user_per_weight:         Objective function values associated with the user weight vector
            - y_user_per_weight:                    Pareto solution associated with the user weight vector 
            - A:                                    Matrix used to define the neighborhood 
            - rhs_ellipsoid:                        RHS of the neighborhood 

        """
        
        if knee_solutions_flag:
          in_val = weight_user
          weight_user = {}  
          weight_user[0] = in_val
              
        ## Determine the Pareto solution for a user-defined weight vector
        f_value_dict_user_per_weight = {}
        y_user_per_weight = {}
        for i in range(len(weight_user)):
            f_value_dict_user, y_user = self.weighted_sum_method(y_init, weight_user[i])
            f_value_dict_user_per_weight[i] = f_value_dict_user
            y_user_per_weight[i] = y_user
        ###############################

        ## Determine the matrix used to define the neighborhood          
        if self.neighborhood_type == 0:
            # Spherical neighborhood
            A = np.eye(self.func.prob.num_obj)   
        else:
            ## Compute the Jacobian of the vector function (f_1,...,f_q) at y wrt the weights
            if self.func.prob.constrained:
                #---------------------------------------------#
                #-------------- Constrained Case -------------#
                #---------------------------------------------#
                jacob_F_lam = self.func.jacob_F_lam_constr(weight_user[0], y_user_per_weight[0])
                
            else:
                #---------------------------------------------#
                #------------- Unconstrained Case ------------#
                #---------------------------------------------#
                jacob_F_lam = self.func.jacob_F_lam(weight_user[0], y_user_per_weight[0])

            if self.iprint >= 3:
                print('\nnabla Fbar: ',jacob_F_lam,' \nPinv: ',np.linalg.pinv(jacob_F_lam),' \nNorm: ',np.linalg.norm(jacob_F_lam),' cond: ',np.linalg.cond(jacob_F_lam),' \nEig: ',np.linalg.eig(jacob_F_lam),'\n')

            ## Ellipsoidal neighborhood
            if self.neighborhood_type == 1:
                A = np.linalg.pinv(jacob_F_lam)  
            ## Cassini oval neighborhood
            elif self.neighborhood_type == 2:                
                A = jacob_F_lam     
        ###############################  
        
        ellipsoid_values_list = []
        
        ## Determine the center of the neighborhood
        if knee_solutions_flag: 
            center = weight_user[0]
        else:
            center = weight_user[0]
                
        ## This is to compute the RHS of the neighborhoods. When fixed RHS doesn't work, we set the RHS to the average of the objective function values in the neighborhood
        for lam in simplex_set_discr.T:
            if self.neighborhood_type == 0 or self.neighborhood_type == 1:
                ellipsoid_value = np.linalg.norm(np.matmul(A,lam.reshape(-1,1) - center))
                ellipsoid_values_list.append(ellipsoid_value)
            elif self.neighborhood_type == 2:
                ellipsoid_value = np.linalg.norm(np.matmul(A,lam.reshape(-1,1) - center))/np.linalg.norm(lam.reshape(-1,1) - center)**2
                ellipsoid_values_list.append(ellipsoid_value)
            # print('Ellipsoid values ',ellipsoid_value)

        if self.neighborhood_type == 0 or self.neighborhood_type == 1:
            constant = 0.4
        elif self.neighborhood_type == 2:
            constant = 1.5  

        rhs_ellipsoid_reference = np.mean(ellipsoid_values_list)*constant
        
        if self.iprint >= 1:   
            print('\nEllipsoid RHS Reference: ',rhs_ellipsoid_reference,'\n')           

        if self.neighborhood_type == 0:
            if self.func.name_prob_to_run == "ZLT1":
                rhs_ellipsoid = 0.40 #ZLT1 
            elif self.func.name_prob_to_run == "GRV1":
                rhs_ellipsoid = 0.30 #GRV1 
            elif self.func.name_prob_to_run == "VFM1":
                rhs_ellipsoid = 0.23 #VFM1  
            elif self.func.name_prob_to_run == "ZLT1q":
                rhs_ellipsoid = 0.28 #ZLT1q 
            else:
                rhs_ellipsoid = rhs_ellipsoid_reference
                
        elif self.neighborhood_type == 1:
            if self.func.name_prob_to_run == "ZLT1":
                rhs_ellipsoid = 0.10 #ZLT1
            elif self.func.name_prob_to_run == "GRV1":
                rhs_ellipsoid = 0.10 #GRV1  
            elif self.func.name_prob_to_run == "VFM1":
                rhs_ellipsoid = 0.10 #VFM1  
            elif self.func.name_prob_to_run == "ZLT1q":
                rhs_ellipsoid = 0.10 #ZLT1q    
            else:
                rhs_ellipsoid = rhs_ellipsoid_reference
        
        elif self.neighborhood_type == 2:
            if self.func.name_prob_to_run == "ZLT1":
                rhs_ellipsoid = 7 #ZLT1
            elif self.func.name_prob_to_run == "GRV1":
                rhs_ellipsoid = 10 #GRV1  
            elif self.func.name_prob_to_run == "VFM1":
                rhs_ellipsoid = 13 #VFM1  
            elif self.func.name_prob_to_run == "ZLT1q":
                rhs_ellipsoid = 8.5 #ZLT1q 
            else:
                rhs_ellipsoid = rhs_ellipsoid_reference
        ##############    

        ## Sample the weight vectors from the simplex set uniformly within the neighborhood 
        ## Store the weight vectors that are in the neighborhood in simplex_set_discr_neighborhood
        simplex_set_discr_neighborhood = np.empty((self.func.prob.num_obj, 0))
            
        for lam in simplex_set_discr.T:    
            if self.neighborhood_type == 0 or self.neighborhood_type == 1:
                if np.linalg.norm(np.matmul(A,lam.reshape(-1,1) - center)) <= rhs_ellipsoid:               
                    simplex_set_discr_neighborhood = np.hstack((simplex_set_discr_neighborhood,lam.reshape(-1,1)))
            elif self.neighborhood_type == 2:
                if np.linalg.norm(np.matmul(A,lam.reshape(-1,1) - center))/np.linalg.norm(lam.reshape(-1,1) - center)**2 >= rhs_ellipsoid:               
                    simplex_set_discr_neighborhood = np.hstack((simplex_set_discr_neighborhood,lam.reshape(-1,1)))

        if self.iprint >= 1:  
            print('\n***Cardinality: Neighborhood: ',simplex_set_discr_neighborhood.shape[1],', Simplex Set: ',simplex_set_discr.shape[1],', Percentage: ',simplex_set_discr_neighborhood.shape[1]/simplex_set_discr.shape[1],'\n')
        ##############   

        ## Compute the Pareto solutions associated with the weight vectors in the neighborhood
        f_value_neighborhood_list_dict = {}
        y_neighborhood_list  = []
        
        ## Create an empty list for each objective
        for m in range(self.func.prob.num_obj):
            f_value_neighborhood_list_dict[m] = []
            
        ## Apply the weighted-sum method for each weight vector in the neighborhood
        ## and store the objective function values and Pareto solutions
        for lam in simplex_set_discr_neighborhood.T:
           out = self.weighted_sum_method(y_init, lam)
           
           for m in range(self.func.prob.num_obj):
               f_value_neighborhood_list_dict[m].append(out[0][m])
               y_neighborhood_list.append(out[1])
        ##############  

        return simplex_set_discr_neighborhood, f_value_neighborhood_list_dict, y_neighborhood_list, \
                f_value_dict_user_per_weight, y_user_per_weight, A, rhs_ellipsoid    
    

    def compute_knee_solutions(self, y_init, initial_guess, simplex_set_discr, f_value_list_dict): 

        # Compute the projection of a vector onto the simplex
        # https://gist.github.com/mblondel/6f3b7aaad90606b98f71
        # License: BSD
        # Author: Mathieu Blondel
        def to_simplex(V, z=1, axis=None):
            """
            Projection of x onto the simplex, scaled by z:
                P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
            z: float or array
                If array, len(z) must be compatible with V
            axis: None or int
                axis=None: project V by P(V.ravel(); z)
                axis=1: project each V[i] by P(V[i]; z[i])
                axis=0: project each V[:, j] by P(V[:, j]; z[j])
            """
            V = np.array(V)
            if axis == 1:
                n_features = V.shape[1]
                U = np.sort(V, axis=1)[:, ::-1]
                z = np.ones(len(V)) * z
                cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
                ind = np.arange(n_features) + 1
                cond = U - cssv / ind > 0
                rho = np.count_nonzero(cond, axis=1)
                theta = cssv[np.arange(len(V)), rho - 1] / rho
                return np.maximum(V - theta[:, np.newaxis], 0)
        
            elif axis == 0:
                return to_simplex(V.T, z, axis=1).T
        
            else:
                V = V.ravel().reshape(1, -1)
                return to_simplex(V, z, axis=1).ravel()
            

        weight_iterates_list = []
        obj_funct_values_list = []
        metric_values_list = []

        ## Global variables to keep track of iterations and function evaluations
        iter_counter = [0]
        feval_counter = [0]
                    
        ## Define the objective function of the snee optimization problem to find knee solutions
        def objective_function(weight_var): 
            # Global feval_counter
            feval_counter[0] += 1

            weight_var = to_simplex(weight_var)
            _, y_user = self.weighted_sum_method(y_init, weight_var)
            
            # Compute the Jacobian of the vector function (f_1,...,f_q) at y wrt the weights
            if self.func.prob.constrained:
                #---------------------------------------------#
                #-------------- Constrained Case -------------#
                #---------------------------------------------#
                jacob_F_lam = self.func.jacob_F_lam_constr(weight_var, y_user)
            else:    
                #---------------------------------------------#
                #------------- Unconstrained Case ------------#
                #---------------------------------------------#
                jacob_F_lam = self.func.jacob_F_lam(weight_var, y_user)
                
            upper_bound = -np.inf    

            for j in range(self.func.prob.num_obj):
                for l in range(self.func.prob.num_obj): 
                    if j == l: # Skip cases where j equals l or k equals l
                        continue
                    value = np.linalg.norm(jacob_F_lam[:, j])/max(np.finfo(float).eps,np.linalg.norm(jacob_F_lam[:, l]))
                    if value >= upper_bound:
                        upper_bound = value
            value = upper_bound                   
        
            return value
        
        ## Define the callback function to store points
        def callback(v):
            # global iter_counter
            # global feval_counter
            iter_counter[0] += 1
            if self.iprint >= 2: 
                print(f"\n---Iteration: {iter_counter[0]}, Feval: {feval_counter[0]}, Weights: {to_simplex(v)}")
            weight_iterates_list.append(to_simplex(v))
            obj_func_val = objective_function(v)
            obj_funct_values_list.append(obj_func_val)
            feval_counter[0] -= 1
            
            if True:            
                _, f_value_neighborhood_list_dict, _, \
                    _, _, _, _ = self.compute_mostchanging_subfronts(to_simplex(v).reshape(-1,1),y_init,simplex_set_discr,knee_solutions_flag=True,iprint=True)
                metric = self.MCM(f_value_list_dict,f_value_neighborhood_list_dict)
                if self.iprint >= 2: 
                    print('\nMCF: ',obj_func_val,' MCM: ',metric,'\n')
                metric_values_list.append(metric)            

        ## Update lists with starting point
        weight_iterates_list.append(to_simplex(initial_guess.flatten()))
        obj_func_val = objective_function(to_simplex(initial_guess.flatten()))
        obj_funct_values_list.append(obj_func_val)            
        _, f_value_neighborhood_list_dict, _, \
            _, _, _, _ = self.compute_mostchanging_subfronts(to_simplex(initial_guess.flatten()).reshape(-1,1),y_init,simplex_set_discr,knee_solutions_flag=True,iprint=True)
        metric = self.MCM(f_value_list_dict,f_value_neighborhood_list_dict)
        if self.iprint >= 2: 
            print('\nInitial MCF: ',obj_func_val,' Initial MCM: ',metric,'\n')
        metric_values_list.append(metric) 
        ########
    
        ## Perform the optimization
        ## Nelder-Mead
        if self.algo_snee == "NM":
            # Set options for the optimizer
            options = {
                'maxfev': None, 
                'maxiter': None, 
            }  
            result = minimize(objective_function, initial_guess.flatten(), constraints=(), method='Nelder-Mead', callback=callback, options=options)
        
        ## DIRECT
        elif self.algo_snee == "DIRECT":
            # Set options for the optimizer
            options = {
                'eps': 0.0001,
                'maxfun': None, 
                'maxiter': 1000, 
                'locally_biased': True,
                'f_min': -np.inf,
                'f_min_rtol': 0.0001,
                'vol_tol': 1e-16,
                'len_tol': 1e-06, 
            }
            bounds = Bounds(np.zeros(self.func.prob.num_obj), np.ones(self.func.prob.num_obj))        
            result = direct(objective_function, bounds, eps=options['eps'], maxfun=options['maxfun'], 
                maxiter=options['maxiter'], locally_biased=options['locally_biased'], 
                f_min=options['f_min'], f_min_rtol=options['f_min_rtol'], 
                vol_tol=options['vol_tol'], len_tol=options['len_tol'], callback=callback) 

        value_test_obj_list = {}            
        
        if self.iprint >= 2: 
            print("\nsnee Optimization Status (0 means success): ", result.status)
            print("result.nfev",result.nfev)

        # Convert the result to the simplex set using the transformation
        weight_var_optimized = to_simplex(result.x)
            
        return weight_var_optimized, weight_iterates_list, obj_funct_values_list, metric_values_list, value_test_obj_list


    def MCM(self,dict1,dict2):
        'Calculates the MCM'
        # from dictionaries (where each item is a list of coordinate values) to lists of points
        X = np.vstack(list(dict1.values()))
        Y = np.vstack(list(dict2.values()))
        if Y.size == 0:
            out = 0
        else:
            out = np.prod(np.abs(np.max(Y,axis=1) - np.min(Y,axis=1))/np.abs(np.max(X,axis=1) - np.min(X,axis=1)))
        return out
    
  

  










     
  










  
    
    
    
    
    
