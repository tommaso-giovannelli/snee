import os
import numpy as np
import random
import torch
from sklearn.datasets import make_spd_matrix
from scipy.sparse.linalg import cg, LinearOperator
from scipy.optimize import approx_fprime


class SyntheticMultiobjProblem:
    """
    Class used to define the following synthetic multi-objective optimization problem:
        min_{y \in Y} F(y), where Y represents the set of feasible points satisfying the constraints. If no constraints are present, Y = R^n.
        
        where F(y) can be:
            - ZLT1 (3 objs)
            - GRV1 (3 objs)
            - VFM1 (3 objs)
            - ZLT1q (q objs)
            - GRV2 (2 objs)
            - DAS1 (2 objs) 
            - DO2DK (2 obj) 
            - VFM1constr (3 objs)

    Attributes
        prob:                         Class representing the problem for which we want to determine most-changing sub-fronts and knee solutions
        name_prob_to_run (str):       A string representing the name of the problem for which we want to determine most-changing sub-fronts and knee solutions
        seed (int, optional):         The seed used for the experiments (default 42)
    """
    
    def __init__(self, name_prob_to_run, seed=42):
        
        self.seed = seed
        
        self.set_seed(self.seed)
        
        if name_prob_to_run == "ZLT1":
            self.prob = ZLT1()
            self.name_prob_to_run = "ZLT1"

        elif name_prob_to_run == "GRV1":
            self.prob = GRV1()
            self.name_prob_to_run = "GRV1"

        elif name_prob_to_run == "VFM1":
            self.prob = VFM1()
            self.name_prob_to_run = "VFM1"

        elif name_prob_to_run == "ZLT1q":
            self.prob = ZLT1q()
            self.name_prob_to_run = "ZLT1q"
            
        elif name_prob_to_run == "GRV2":
            self.prob = GRV2()
            self.name_prob_to_run = "GRV2"

        elif name_prob_to_run == "DAS1":
            self.prob = DAS1()
            self.name_prob_to_run = "DAS1"
            
        elif name_prob_to_run == "DO2DK":
            self.prob = DO2DK()
            self.name_prob_to_run = "DO2DK"

        elif name_prob_to_run == "DO2DKtight":
            self.prob = DO2DK(tight_flag=True)
            self.name_prob_to_run = "DO2DK"
            
        elif name_prob_to_run == "VFM1constr":
            self.prob = VFM1constr()
            self.name_prob_to_run = "VFM1constr"
            
        
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
    

    def f_weighted(self, lam, y):
        """
        The weighted objective function: f = sum_{j=1}^{q} f_j 
        
        Args:
            lam --> Weights in the simplex set
        """
        out = 0
        
        for m in range(self.prob.num_obj):
            out += lam[m]*self.prob.f_dict[m](y)
            
        return (out).squeeze()


    def grad_f_weighted_lam(self, lam, y):
        """
        The gradient of the weighted objective function wrt the weights
        
        Args:
            lam --> Weights in the simplex set
    	"""
        out = np.empty((0,1))
        
        for m in range(self.prob.num_obj):
            out = np.vstack((out,self.prob.f_dict[m](y)))
        return out


    def grad_f_weighted_vars(self, lam, y):
        """
        The gradient of the weighted objective function wrt the variables
        
        Args:
            lam --> Weights in the simplex set
    	"""
        out = np.zeros((self.prob.dim,1))
        
        for m in range(self.prob.num_obj):
            out += lam[m]*self.prob.grad_f_dict[m](y)
        return out
    
    
    def hess_f_weighted_vars(self, lam, y):
        """
        The Hessian of the weighted objective function wrt the variables
        
        Args:
            lam --> Weights in the simplex set
    	"""
        out = np.zeros((self.prob.dim,self.prob.dim))
        
        for m in range(self.prob.num_obj):
            out += lam[m]*self.prob.hess_f_dict[m](y)
        return out


    def jacob_y_lam(self, lam, y):
        """
        The Jacobian of the Pareto point y wrt the weights
        
        Args:
            lam --> Weights in the simplex set
    	"""
        aux = np.empty((0,self.prob.dim))
        
        for m in range(self.prob.num_obj):
            aux = np.vstack((aux,self.prob.grad_f_dict[m](y).T))
        
        out = -np.matmul(aux, np.linalg.inv(self.hess_f_weighted_vars(lam, y)))
        return out
    

    def jacob_F_y(self, lam, y):
        """
        The Jacobian of the vector function (f_1,...,f_q) at y wrt to y
        
        Args:
            lam --> Weights in the simplex set
    	"""
        out = np.empty((self.prob.dim,0))

        for m in range(self.prob.num_obj):
            out = np.hstack((out,self.prob.grad_f_dict[m](y)))
            
        return out
    
    
    def jacob_F_lam(self, lam, y):
        """
        The Jacobian of the vector function (f_1,...,f_q) at y wrt the weights
        
        Args:
            lam --> Weights in the simplex set
    	"""
        return np.matmul(self.jacob_y_lam(lam, y), self.jacob_F_y(lam, y))


    ## Constrained case
    def solve_kkt_system(self, lam, y):
      """
      Solves the KKT system using least-squares and CG to determine the optimal Lagrange multiplier vector

      """
      # This code is based on the assumption that there is at least one inequality constraint
      if self.prob.num_eq_constr==0:
          jacob_y = self.prob.inequality_constraints_jacob(y)
          inconstr_vec = self.prob.inequality_constraints(y)

          # Gradient of objective function wrt y
          grad_y = self.grad_f_weighted_vars(lam, y) 

          lagr_mul = np.ones((self.prob.num_ineq_constr, 1)) #None

          def mv(v):
              out = np.matmul(jacob_y.T, np.matmul(jacob_y, v.reshape(-1,1))) + np.multiply(inconstr_vec**2, v.reshape(-1,1)) 
              return out  
            
          G = LinearOperator((self.prob.num_ineq_constr,self.prob.num_ineq_constr), matvec=mv)
            
          lagr_mul, exit_code = cg(G,-np.matmul(jacob_y.T,grad_y), x0=lagr_mul, maxiter=100) #tol=1e-4 #maxiter=100

      else:
          injacob_y = self.prob.inequality_constraints_jacob(y)
          eqjacob_y = self.prob.equality_constraints_jacob(y)
          inconstr_vec = self.prob.inequality_constraints(y)
    
          # Gradient of objective function wrt y
          grad_fweighted_y = self.grad_f_weighted_vars(lam, y) 
    
          lagr_mul = np.ones((self.prob.num_ineq_constr+self.prob.num_eq_constr, 1)) 
    
          def mv(v):
                # Slice v into components
                v_ineq = v[:self.prob.num_ineq_constr].reshape(-1, 1)  # Components for inequality constraints
                v_eq = v[self.prob.num_ineq_constr:self.prob.num_ineq_constr + self.prob.num_eq_constr].reshape(-1, 1)  # Components for equality constraints
            
                # Compute the sub-blocks
                out1 = (
                    np.matmul(injacob_y.T, np.matmul(injacob_y, v_ineq)) +
                    np.multiply(inconstr_vec**2, v_ineq)
                )
                out2 = np.matmul(injacob_y.T, np.matmul(eqjacob_y, v_eq))
                out3 = np.matmul(eqjacob_y.T, np.matmul(injacob_y, v_ineq))
                out4 = np.matmul(eqjacob_y.T, np.matmul(eqjacob_y, v_eq))
            
                # Construct the 2x2 numpy matrix
                result = np.vstack([out1 + out2, out3 + out4])
                
                return result  
            
          G = LinearOperator((self.prob.num_ineq_constr+self.prob.num_eq_constr,self.prob.num_ineq_constr+self.prob.num_eq_constr), matvec=mv)
    
          component1 = -np.matmul(injacob_y.T, grad_fweighted_y)
          component2 = -np.matmul(eqjacob_y.T, grad_fweighted_y)
    
          lagr_mul, exit_code = cg(G, np.vstack([component1, component2]), x0=lagr_mul, maxiter=100) 
      
      lagr_mul = np.reshape(lagr_mul, (-1,1))
      return lagr_mul
    
    
    def jacob_F_lam_constr(self, lam, y):
        """
        The Jacobian of the vector function (f_1,...,f_q) at y wrt the weights in the constrained case
        
        Args:
            lam --> Weights in the simplex set
    	"""
        return np.matmul(self.jacob_y_lam_constr(lam, y), self.jacob_F_y(lam, y))
    
    
    def jacob_y_lam_constr(self, lam, y):
        """
        The Jacobian of the Pareto point y wrt the weights
        
        Args:
            lam --> Weights in the simplex set
    	"""
        I = np.eye(self.prob.dim)
        null_matrix = np.zeros((self.prob.dim, self.prob.num_ineq_constr+self.prob.num_eq_constr))
        L = np.vstack((I, null_matrix.T))
        
        lagr_mul = self.solve_kkt_system(lam, y)
        
        out = -np.matmul(self.jacob_kkt_lam(y, lam, lagr_mul), np.matmul(np.linalg.inv(self.jacob_kkt_w(y, lam, lagr_mul)), L))
        return out
        

    def hess_lagr_func_y_y(self, y, lam, lagr_mul):
        """
        Hessian of the Lagrangian function
        """
        # This code is based on the assumption that there is at least one inequality constraint
        if self.prob.num_eq_constr==0:
            hess_yy = self.hess_f_weighted_vars(lam, y)      
            
            hess_constr_yy_list = self.prob.inequality_constraints_hess(y)
            
            for i in range(self.prob.num_ineq_constr):
                hess_yy = hess_yy + hess_constr_yy_list[i]*lagr_mul[i]

        else:
            hess_yy = self.hess_f_weighted_vars(lam, y)      
            
            hess_inconstr_yy_list = self.prob.inequality_constraints_hess(y)
            hess_eqconstr_yy_list = self.prob.equality_constraints_hess(y)
            
            v_ineq = lagr_mul[:self.prob.num_ineq_constr].reshape(-1, 1)  # Components for inequality constraints
            v_eq = lagr_mul[self.prob.num_ineq_constr:self.prob.num_ineq_constr + self.prob.num_eq_constr].reshape(-1, 1)  # Components for equality constraints
            
            for i in range(self.prob.num_ineq_constr):
                hess_yy = hess_yy + hess_inconstr_yy_list[i]*v_ineq[i] 
    
            for i in range(self.prob.num_eq_constr):
                hess_yy = hess_yy + hess_eqconstr_yy_list[i]*v_eq[i]

        return hess_yy


    def hess_lagr_func_y_lam(self, y, lam, lagr_mul):
        """
        Hessian of the Lagrangian function
        """
        hess_y_lam = self.jacob_F_y(lam, y)
        return hess_y_lam + np.zeros((self.prob.dim,self.prob.num_obj))


    def jacob_kkt_w(self, y, lam, lagr_mul):
        """
        Jacobian of the KKT system wrt w
        """
        # This code is based on the assumption that there is at least one inequality constraint
        if self.prob.num_eq_constr==0:
            jacob_y = self.prob.inequality_constraints_jacob(y)
                        
            out_1 = np.concatenate((self.hess_lagr_func_y_y(y, lam, lagr_mul), lagr_mul.T*jacob_y), axis=1)            
            if self.prob.__class__.__name__ == 'DAS1':    
                ### DAS1 without inequality constraints
                out_2 = np.concatenate((jacob_y.T,np.diag(np.atleast_1d(np.squeeze(self.prob.inequality_constraints(y))))), axis=1)
            else:
                out_2 = np.concatenate((jacob_y.T,np.diag(np.squeeze(self.prob.inequality_constraints(y)))), axis=1)
            out = np.concatenate((out_1,out_2), axis=0)   

        else:
            injacob_y = self.prob.inequality_constraints_jacob(y)
            eqjacob_y = self.prob.equality_constraints_jacob(y)
            
            v_ineq = lagr_mul[:self.prob.num_ineq_constr].reshape(-1, 1)  # Components for inequality constraints
                    
            out_1 = np.concatenate((self.hess_lagr_func_y_y(y, lam, lagr_mul), v_ineq.T*injacob_y, eqjacob_y), axis=1)
            out_2 = np.concatenate((injacob_y.T,np.diag(self.prob.inequality_constraints(y).flatten()),np.zeros((self.prob.num_ineq_constr,self.prob.num_eq_constr))), axis=1)
            out_3 = np.concatenate((eqjacob_y.T,np.zeros((self.prob.num_eq_constr,self.prob.num_ineq_constr)),np.zeros((self.prob.num_eq_constr,self.prob.num_eq_constr))), axis=1)
            out = np.vstack([out_1, out_2, out_3])
        return out


    def jacob_kkt_lam(self, y, lam, lagr_mul):
        """
        Jacobian of the KKT system wrt lam
        """
        out = np.concatenate((self.hess_lagr_func_y_lam(y, lam, lagr_mul).T, np.zeros((self.prob.num_obj,self.prob.num_ineq_constr)), np.zeros((self.prob.num_obj,self.prob.num_eq_constr))), axis=1) 
        return out    


    ## Auxiliary functions
    def check_kkt_system(self, lam, y):
      """
      Check whether the equality part of the KKT system is satisfied

      """
      # This code is based on the assumption that there is at least one inequality constraint
      if self.prob.num_eq_constr==0:
          injacob_y = self.prob.inequality_constraints_jacob(y)
          grad_fweighted_y = self.grad_f_weighted_vars(lam, y) 
          inconstr_vec = self.prob.inequality_constraints(y)
          
          lagr_mul = self.solve_kkt_system(lam, y)
    
          v_ineq = lagr_mul[:self.prob.num_ineq_constr].reshape(-1, 1)  # Components for inequality constraints
    
          return np.vstack([grad_fweighted_y + np.matmul(injacob_y, v_ineq), v_ineq * inconstr_vec])          
          
      else:
          injacob_y = self.prob.inequality_constraints_jacob(y)
          eqjacob_y = self.prob.equality_constraints_jacob(y)
          grad_fweighted_y = self.grad_f_weighted_vars(lam, y) 
          inconstr_vec = self.prob.inequality_constraints(y)
          eqconstr_vec = self.prob.equality_constraints(y)
          
          lagr_mul = self.solve_kkt_system(lam, y)
    
          v_ineq = lagr_mul[:self.prob.num_ineq_constr].reshape(-1, 1)  # Components for inequality constraints
          v_eq = lagr_mul[self.prob.num_ineq_constr:self.prob.num_ineq_constr + self.prob.num_eq_constr].reshape(-1, 1)  # Components for equality constraints
    
          return np.vstack([grad_fweighted_y + np.matmul(injacob_y, v_ineq) + np.matmul(eqjacob_y, v_eq), v_ineq * inconstr_vec, eqconstr_vec])


    def check_gradient(self):
        def f_test(y):
            return self.prob.f_2(y)
        
        y = np.random.rand(self.prob.dim)  
        grad_approx = approx_fprime(y, f_test, epsilon=1e-8)
        grad_computed = self.prob.grad_f2(y).flatten()
        
        print("Approximated Gradient:", grad_approx)
        print("Computed Gradient:", grad_computed)
        print("Difference:", grad_approx - grad_computed)


    def check_hessian(self):
        def compute_hessian(f, y, epsilon=1e-6):
            """
            Numerically approximate the Hessian matrix of f at point y using central finite differences.
            """
            n = len(y)
            hessian = np.zeros((n, n))
            
            # Compute the second derivatives (Hessian matrix) using central differences
            for i in range(n):
                # Define the perturbation for the i-th direction
                y_pos = y.copy()
                y_neg = y.copy()
                y_pos[i] += epsilon
                y_neg[i] -= epsilon
                
                # Approximate second derivatives
                hessian[:, i] = (approx_fprime(y_pos, f, epsilon) - approx_fprime(y_neg, f, epsilon)) / (2 * epsilon)
            
            return hessian
        
        y = np.random.rand(self.prob.dim)  # Replace 5 with the dimensionality of y
        
        # Define the function to compute the objective
        def f_test(y):
            return self.prob.f_1(y)
        
        hess_approx = compute_hessian(f_test, y)
        hess_computed = self.prob.hess_f1(y)
        
        print("Approximated Hessian:\n", hess_approx)
        print("Computed Hessian:\n", hess_computed)
        print("Difference:\n", hess_approx - hess_computed)   


class ZLT1:

    def __init__(self):
        
        self.dim = 3 
        self.num_obj = 3
        
        self.f_dict = {}
        self.f_dict[0] = self.f_1
        self.f_dict[1] = self.f_2
        self.f_dict[2] = self.f_3
        
        self.grad_f_dict = {}
        self.grad_f_dict[0] = self.grad_f1
        self.grad_f_dict[1] = self.grad_f2
        self.grad_f_dict[2] = self.grad_f3

        self.hess_f_dict = {}
        self.hess_f_dict[0] = self.hess_f1
        self.hess_f_dict[1] = self.hess_f2
        self.hess_f_dict[2] = self.hess_f3

        self.weight_user = {0: np.array([[0.8],[0.1],[0.1]])}  
        
        self.num_pareto_points = 50

        self.constrained = False
        

    def f_1(self, y):
        """
        Objective function 1 
    	"""
        aux = np.zeros((self.dim,1))
        aux[0] = -1
        return (np.dot((y+aux).T,y+aux)).squeeze()


    def f_2(self, y):
        """
        Objective function 2 
    	"""
        aux = np.zeros((self.dim,1))
        aux[1] = -1
        return (np.dot((y+aux).T,y+aux)).squeeze()


    def f_3(self, y):
        """
        Objective function 3 
    	"""
        aux = np.zeros((self.dim,1))
        aux[2] = -1
        return (np.dot((y+aux).T,y+aux)).squeeze()
    

    def grad_f1(self, y):
        """
        The gradient of objective function 1 
    	"""
        aux = np.zeros((self.dim,1))
        aux[0] = -1
        return 2*(y+aux)
    
    
    def grad_f2(self, y):
        """
        The gradient of objective function 2 
    	"""
        aux = np.zeros((self.dim,1))
        aux[1] = -1
        return 2*(y+aux)


    def grad_f3(self, y):
        """
        The gradient of objective function 3
    	"""
        aux = np.zeros((self.dim,1))
        aux[2] = -1
        return 2*(y+aux)
    

    def hess_f1(self, y):
        """
        The Hessian of objective function 1 
     	"""
        return 2*np.eye(self.dim)


    def hess_f2(self, y):
        """
        The Hessian of objective function 2 
     	"""
        return 2*np.eye(self.dim)


    def hess_f3(self, y):
        """
        The Hessian of objective function 3 
     	"""
        return 2*np.eye(self.dim) 


class GRV1:
    
    def __init__(self):
        
        self.dim = 2      
        self.num_obj = 3
        
        self.f_dict = {}
        self.f_dict[0] = self.f_1
        self.f_dict[1] = self.f_2
        self.f_dict[2] = self.f_3
        
        self.grad_f_dict = {}
        self.grad_f_dict[0] = self.grad_f1
        self.grad_f_dict[1] = self.grad_f2
        self.grad_f_dict[2] = self.grad_f3

        self.hess_f_dict = {}
        self.hess_f_dict[0] = self.hess_f1
        self.hess_f_dict[1] = self.hess_f2
        self.hess_f_dict[2] = self.hess_f3
        
        self.weight_user = {0: np.array([[0.8],[0.1],[0.1]])} 
        
        half_dim = int(self.dim/2)
        
        original_matrix_1 = make_spd_matrix(self.dim, random_state=0)*20
        # print('original_matrix_1',original_matrix_1)
        
        self.H1 = original_matrix_1[:half_dim, :half_dim]
        self.H3 = original_matrix_1[:half_dim, half_dim:]
        self.H2 = original_matrix_1[half_dim:, half_dim:]

        original_matrix_2 = make_spd_matrix(self.dim, random_state=1)*20
        # print('original_matrix_2',original_matrix_2)
        
        self.H4 = original_matrix_2[:half_dim, :half_dim]
        self.H6 = original_matrix_2[:half_dim, half_dim:]
        self.H5 = original_matrix_2[half_dim:, half_dim:]

        original_matrix_3 = make_spd_matrix(self.dim, random_state=2)*20
        # print('original_matrix_3',original_matrix_3)
        
        self.H7 = original_matrix_3[:half_dim, :half_dim]
        self.H9 = original_matrix_3[:half_dim, half_dim:]
        self.H8 = original_matrix_3[half_dim:, half_dim:]
        
        self.a1 = np.random.uniform(0, -5,(half_dim,1))
        self.a2 = np.random.uniform(0, -5,(half_dim,1))

        self.a3 = np.random.uniform(0, 5,(half_dim,1))
        self.a4 = np.random.uniform(0, 5,(half_dim,1))

        self.a5 = np.random.uniform(0, -5,(half_dim,1))
        self.a6 = np.random.uniform(0, 5,(half_dim,1))
        
        # print('a1',self.a1)
        # print('a2',self.a2)
        # print('a3',self.a3)
        # print('a4',self.a4)
        # print('a5',self.a5)
        # print('a6',self.a6)

        self.num_pareto_points = 50
        
        self.constrained = False
        
        
    def f_1(self, y):
        """
        Objective function 1 
    	"""
        x1 = y[:int(self.dim/2)]
        x2 = y[int(self.dim/2):]
        return (0.5*np.dot(x1.T,np.dot(self.H1,x1)) + 0.5*np.dot(x2.T,np.dot(self.H2,x2)) + np.dot(x1.T,np.dot(self.H3,x2)) + np.dot(self.a1.T,x1) + np.dot(self.a2.T,x2)).squeeze()


    def f_2(self, y):
        """
        Objective function 2 
    	"""
        x1 = y[:int(self.dim/2)]
        x2 = y[int(self.dim/2):]
        return (0.5*np.dot(x1.T,np.dot(self.H4,x1)) + 0.5*np.dot(x2.T,np.dot(self.H5,x2)) + np.dot(x1.T,np.dot(self.H6,x2)) + np.dot(self.a3.T,x1) + np.dot(self.a4.T,x2)).squeeze() 
    

    def f_3(self, y):
        """
        Objective function 3 
    	"""
        x1 = y[:int(self.dim/2)]
        x2 = y[int(self.dim/2):]
        return (0.5*np.dot(x1.T,np.dot(self.H7,x1)) + 0.5*np.dot(x2.T,np.dot(self.H8,x2)) + np.dot(x1.T,np.dot(self.H9,x2)) + np.dot(self.a5.T,x1) + np.dot(self.a6.T,x2)).squeeze() 


    def grad_f1(self, y):
        """
        The gradient of objective function 1 
    	"""
        x1 = y[:int(self.dim/2)]
        x2 = y[int(self.dim/2):]
        out_x1 = np.dot(self.H1,x1) + np.dot(self.H3,x2) + self.a1
        out_x2 = np.dot(self.H2,x2) + np.dot(self.H3.T,x1) + self.a2
        return np.concatenate((out_x1,out_x2),axis=0)
    
    
    def grad_f2(self, y):
        """
        The gradient of objective function 2 
    	"""
        x1 = y[:int(self.dim/2)]
        x2 = y[int(self.dim/2):]
        out_x1 = np.dot(self.H4,x1) + np.dot(self.H6,x2) + self.a3
        out_x2 = np.dot(self.H5,x2) + np.dot(self.H6.T,x1) + self.a4
        return np.concatenate((out_x1,out_x2),axis=0)  


    def grad_f3(self, y):
        """
        The gradient of objective function 3 
    	"""
        x1 = y[:int(self.dim/2)]
        x2 = y[int(self.dim/2):]
        out_x1 = np.dot(self.H7,x1) + np.dot(self.H9,x2) + self.a5
        out_x2 = np.dot(self.H8,x2) + np.dot(self.H9.T,x1) + self.a6
        return np.concatenate((out_x1,out_x2),axis=0) 
    

    def hess_f1(self, y):
        """
        The Hessian of objective function 1 
     	"""
        out_x1x1 = self.H1
        out_x1x2 = self.H3
        out_x2x2 = self.H2
        return np.concatenate((np.concatenate((out_x1x1, out_x1x2), axis=1), np.concatenate((out_x1x2.T, out_x2x2), axis=1)), axis=0)
        

    def hess_f2(self, y):
        """
        The Hessian of objective function 2 
     	"""
        out_x1x1 = self.H4
        out_x1x2 = self.H6 
        out_x2x2 = self.H5
        return np.concatenate((np.concatenate((out_x1x1, out_x1x2), axis=1), np.concatenate((out_x1x2.T, out_x2x2), axis=1)), axis=0)


    def hess_f3(self, y):
        """
        The Hessian of objective function 3 
     	"""
        out_x1x1 = self.H7
        out_x1x2 = self.H9 
        out_x2x2 = self.H8
        return np.concatenate((np.concatenate((out_x1x1, out_x1x2), axis=1), np.concatenate((out_x1x2.T, out_x2x2), axis=1)), axis=0)


class VFM1:

    def __init__(self):
        
        self.dim = 2 
        self.num_obj = 3
        
        self.f_dict = {}
        self.f_dict[0] = self.f_1
        self.f_dict[1] = self.f_2
        self.f_dict[2] = self.f_3
        
        self.grad_f_dict = {}
        self.grad_f_dict[0] = self.grad_f1
        self.grad_f_dict[1] = self.grad_f2
        self.grad_f_dict[2] = self.grad_f3

        self.hess_f_dict = {}
        self.hess_f_dict[0] = self.hess_f1
        self.hess_f_dict[1] = self.hess_f2
        self.hess_f_dict[2] = self.hess_f3

        self.weight_user = {0: np.array([[0.4],[0.2],[0.4]])} 

        self.num_pareto_points = 50 

        self.constrained = False
        

    def f_1(self, y):
        """
        Objective function 1 
    	"""
        aux = np.zeros((self.dim,1))
        aux[1] = -1
        return (np.dot((y+aux).T,y+aux)).squeeze()


    def f_2(self, y):
        """
        Objective function 2 
    	"""
        aux = np.zeros((self.dim,1))
        aux[1] = 1
        return (np.dot((y+aux).T,y+aux) + 1).squeeze()


    def f_3(self, y):
        """
        Objective function 3 
    	"""
        aux = np.zeros((self.dim,1))
        aux[0] = -1
        return (np.dot((y+aux).T,y+aux) + 2).squeeze()
    

    def grad_f1(self, y):
        """
        The gradient of objective function 1 
    	"""
        aux = np.zeros((self.dim,1))
        aux[1] = -1
        return 2*(y+aux)
    
    
    def grad_f2(self, y):
        """
        The gradient of objective function 2 
    	"""
        aux = np.zeros((self.dim,1))
        aux[1] = 1
        return 2*(y+aux)


    def grad_f3(self, y):
        """
        The gradient of objective function 3
    	"""
        aux = np.zeros((self.dim,1))
        aux[0] = -1
        return 2*(y+aux)
    

    def hess_f1(self, y):
        """
        The Hessian of objective function 1 
     	"""
        return 2*np.eye(self.dim)


    def hess_f2(self, y):
        """
        The Hessian of objective function 2 
     	"""
        return 2*np.eye(self.dim)


    def hess_f3(self, y):
        """
        The Hessian of objective function 3 
     	"""
        return 2*np.eye(self.dim) 


class ZLT1q:

    def __init__(self):
        
        self.dim = 5 
        self.num_obj = 5
        
        self.f_dict = {}
        for i in range(self.num_obj):
            self.f_dict[i] = lambda y, i=i: self.f_i(y, i)        
        
        self.grad_f_dict = {}
        for i in range(self.num_obj):
            self.grad_f_dict[i] = lambda y, i=i: self.grad_fi(y, i) 

        self.hess_f_dict = {}
        for i in range(self.num_obj):
            self.hess_f_dict[i] = lambda y, i=i: self.hess_fi(y, i) 

        self.weight_user = {0: np.array([[0.6],[0.1],[0.1],[0.1],[0.1]])}

        self.num_pareto_points = 15

        self.constrained = False
        

    def f_i(self, y, i):
        """
        Objective function i 
    	"""
        aux = np.zeros((self.dim,1))
        aux[i] = -1
        return (np.dot((y+aux).T,y+aux)).squeeze()
    

    def grad_fi(self, y, i):
        """
        The gradient of objective function i 
    	"""
        aux = np.zeros((self.dim,1))
        aux[i] = -1
        return 2*(y+aux)
    

    def hess_fi(self, y, i):
        """
        The Hessian of objective function i
     	"""
        return 2*np.eye(self.dim)
    
    
class GRV2:
    
    def __init__(self):
        
        self.dim = 2        
        self.num_obj = 2
        
        self.f_dict = {}
        self.f_dict[0] = self.f_1
        self.f_dict[1] = self.f_2
        
        self.grad_f_dict = {}
        self.grad_f_dict[0] = self.grad_f1
        self.grad_f_dict[1] = self.grad_f2

        self.hess_f_dict = {}
        self.hess_f_dict[0] = self.hess_f1
        self.hess_f_dict[1] = self.hess_f2

        self.weight_user = {0: np.array([[0.9],[0.1]])}

        self.num_pareto_points = 1000 #2000
        
        self.constrained = False


    def f_1(self, y):
        return (1/self.dim)*np.sum(y**2, axis=0).squeeze() + 0.5 * np.sum(y**4, axis=0).squeeze()

    def f_2(self, y):
        return (1/self.dim)*np.sum((y-2)**2, axis=0).squeeze() + 0.5 * np.sum((y-2)**4, axis=0).squeeze()
    
    def grad_f1(self, y):
        return (2/self.dim)*y + 2 *y**3

    def grad_f2(self, y):
        return (2/self.dim)*(y-2) + 2 * (y-2)**3

    def hess_f1(self, y):
        return np.eye(self.dim)*(2/self.dim) + 6 * np.diag(y**2)

    def hess_f2(self, y):
        return np.eye(self.dim)*(2/self.dim) + 6 * np.diag((y-2)**2)


class DAS1:
    
    def __init__(self):
        
        self.dim = 5      
        self.num_obj = 2
        
        self.f_dict = {}
        self.f_dict[0] = self.f_1
        self.f_dict[1] = self.f_2
        
        self.grad_f_dict = {}
        self.grad_f_dict[0] = self.grad_f1
        self.grad_f_dict[1] = self.grad_f2

        self.hess_f_dict = {}
        self.hess_f_dict[0] = self.hess_f1
        self.hess_f_dict[1] = self.hess_f2
        
        self.weight_user = {0: np.array([[0.4],[0.6]])}   

        self.num_pareto_points = 100
        
        self.num_ineq_constr = 1
        self.num_eq_constr = 2  

        self.constrained = True
        

    def f_1(self, y):
        """
        The objective function 2 
    	"""
        return (y[0]**2 + y[1]**2 + y[2]**2 + y[3]**2 + y[4]**2).squeeze()
     

    def f_2(self, y):
        """
        The objective function 1 
    	"""
        return (3*y[0] + 2*y[1] - y[2]/3 + 0.01*(y[3] - y[4])**3).squeeze()
    

    def grad_f1(self, y):
        """
        The gradient of the objective function 1 
    	"""
        grad = 2 * y        
        return grad.reshape(-1,1)
  
    
    def grad_f2(self, y):
        """
        The gradient of the objective function 2 
    	"""
        grad = np.array([3, 2, -1/3, 0.03*(y[3].item() - y[4].item())**2, -0.03*(y[3].item() - y[4].item())**2])        
        return grad.reshape(-1,1)
    

    def hess_f1(self, y):
        """
        The Hessian of objective function 1 
     	"""
        H = 2 * np.eye(len(y))        
        return H
        

    def hess_f2(self, y):
        """
        The Hessian of objective function 2 
     	"""
        H = np.zeros((len(y), len(y)))  # Initialize a zero matrix
        H[3, 3] = 0.06 * (y[3] - y[4])  # Partial derivative w.r.t y[3], y[3]
        H[3, 4] = -0.06 * (y[3] - y[4])  # Partial derivative w.r.t y[3], y[4]
        H[4, 3] = -0.06 * (y[3] - y[4])  # Partial derivative w.r.t y[4], y[3]
        H[4, 4] = 0.06 * (y[3] - y[4])  # Partial derivative w.r.t y[4], y[4]        
        return H
    
    
    def inequality_constraints(self, y):
        """
        Inequality constraints
     	"""
        return np.array([y[0]**2 + y[1]**2 + y[2]**2 + y[3]**2 + y[4]**2 - 10])      


    def inequality_constraints_jacob(self, y):
        """
        Jacobian of the inequality constraints
        """
        jacobian = np.array([2 * y[0], 2 * y[1], 2 * y[2], 2 * y[3], 2 * y[4]])
        return jacobian.reshape(-1, 1)
    
    
    def inequality_constraints_hess(self, y):
        """
        Hessian of the inequality constraints (as a list of arrays)
        """
        num_constraints = 1
        hessian = 2 * np.eye(self.dim)  
        hessians = [hessian for _ in range(num_constraints)]  # List of Hessians
        return hessians


    def inequality_constraints_hess_with_multipliers(self, y, lagr_mul):
        """
        Hessian of the inequality constraints (as a list of arrays)
        """
        num_constraints = 1
        hessian = 2 * np.eye(self.dim)  

        # Multiply each Hessian by the corresponding Lagrange multiplier
        hessians = []
        for i in range(num_constraints):
            hessians.append(lagr_mul[i] * hessian)

        return np.sum(hessians, axis=0)
    
    
    def equality_constraints(self, y):
        """
        Equality constraints
     	"""
        constr1 = y[0] + 2*y[1] - y[2] - 0.5*y[3] + y[4] - 2
        constr2 = 4*y[0] - 2*y[1] + 0.8*y[2] + 0.6*y[3] + 0.5*y[4]**2
        return np.vstack([constr1, constr2])    


    def equality_constraints_jacob(self, y):
        """
        Jacobian of the equality constraints
        """
        # Each row corresponds to the gradient of a constraint
        jacobian = np.array([
            [1, 2, -1, -0.5, 1],               
            [4, -2, 0.8, 0.6, y[4].item()]           
        ])
        return jacobian.T
    
    
    def equality_constraints_hess(self, y):
        """
        Hessian of the equality constraints
        """
        hessians = []

        hessians.append(np.zeros((self.dim, self.dim)))

        # Hessian for g2:
        hess_g2 = np.zeros((self.dim, self.dim))
        hess_g2[4, 4] = 1  # Second derivative w.r.t. y[4]^2 is 1
        hessians.append(hess_g2)

        return hessians


    def equality_constraints_hess_with_multipliers(self, y, lagr_mul):
        """
        Hessian of the equality constraints
        """
        hessians = []

        # Hessian for g1: All second derivatives are 0 (linear function)
        hessians.append(np.zeros((self.dim, self.dim)))

        # Hessian for g2:
        hess_g2 = np.zeros((self.dim, self.dim))
        hess_g2[4, 4] = 1  # Second derivative w.r.t. y[4]^2 is 1
        hessians.append(hess_g2*lagr_mul[1])

        return np.sum(hessians, axis=0)    


class DO2DK:
    
    def __init__(self, tight_flag):
        
        self.dim = 30 #30      
        self.num_obj = 2
        
        self.f_dict = {}
        self.f_dict[0] = self.f_1
        self.f_dict[1] = self.f_2
        
        self.grad_f_dict = {}
        self.grad_f_dict[0] = self.grad_f1
        self.grad_f_dict[1] = self.grad_f2

        self.hess_f_dict = {}
        self.hess_f_dict[0] = self.hess_f1
        self.hess_f_dict[1] = self.hess_f2
        
        self.weight_user = {0: np.array([[0.2],[0.8]])}   
        
        self.s = 0
        self.K = 1

        self.tight_flag = tight_flag
        
        self.num_pareto_points = 100
        
        self.num_ineq_constr = self.dim*2
        self.num_eq_constr = 0

        self.constrained = True
        

    def g(self, y):
        return 1 + (9 / (len(y) - 1)) * np.sum(y[1:])
    
    
    def r(self, y1):
        return 5 + 10 * (y1 - 0.5) ** 2 + (1 / self.K) * np.cos(2 * self.K * np.pi * y1) * 2 ** (self.s / 2)


    def f_1(self, y):
        """
        The objective function 2 
    	"""
        g_val = self.g(y)
        r_val = self.r(y[0])
        term1 = np.sin(np.pi * y[0] / (2 ** (self.s + 1)) + (1 + (2 ** self.s - 1) / (2 ** (self.s + 2))) * np.pi)
        return (g_val * r_val * (term1 + 1) + 0*y[0]**2).squeeze()
    

    def f_2(self, y):
        """
        The objective function 1 
    	"""
        g_val = self.g(y)
        r_val = self.r(y[0])
        term2 = np.cos(np.pi * y[0] / 2 + np.pi)
        return (g_val * r_val * (term2 + 1) + 0*y[0]**2).squeeze()
    

    def grad_f1(self, y):
        """
        The gradient of the objective function 1 
    	"""
        n = len(y)
        y1 = y[0]
        
        grad = np.zeros(n)
        
        # Gradient with respect to x1
        term1 = self.g(y) * (20*(y1 - 0.5) - 2*self.K*np.pi*np.sin(2*self.K*np.pi*y1) * 2**(self.s/2)/self.K) * (np.sin(np.pi * y1 / 2**(self.s+1) + (1 + (2**self.s - 1) / 2**(self.s+2)) * np.pi) + 1)
        term2 = self.g(y) * self.r(y1) * (np.pi/2**(self.s+1)) * np.cos(np.pi * y1 / 2**(self.s+1) + (1 + (2**self.s - 1) / 2**(self.s+2)) * np.pi)
        grad[0] = term1 + term2 + 0*2*y[0]
        
        # Gradient with respect to x2, x3, ..., xn
        grad[1:] = (9/(n-1)) * self.r(y1) * (np.sin(np.pi * y1 / 2**(self.s+1) + (1 + (2**self.s - 1) / 2**(self.s+2)) * np.pi) + 1)
        
        return grad.reshape(-1,1)
  
    
    def grad_f2(self, y):
        """
        The gradient of the objective function 2 
    	"""
        n = len(y)
        y1 = y[0]
        
        grad = np.zeros(n)
        
        # Gradient with respect to x1
        term1 = self.g(y) * (20*(y1 - 0.5) - 2*self.K*np.pi*np.sin(2*self.K*np.pi*y1) * 2**(self.s/2)/self.K) * (np.cos(np.pi * y1 / 2 + np.pi) + 1)
        term2 = -self.g(y) * self.r(y1) * (np.pi/2) * np.sin(np.pi * y1 / 2 + np.pi)
        grad[0] = term1 + term2 + 0*2*y[0]
        
        # Gradient with respect to x2, x3, ..., xn
        grad[1:] = (9/(n-1)) * self.r(y1) * (np.cos(np.pi * y1 / 2 + np.pi) + 1)
        
        return grad.reshape(-1,1)
    

    def hess_f1(self, y):
        """
        The Hessian of objective function 1 
     	"""
        n = len(y)
        y1 = y[0]
        
        # Initialize Hessian matrix
        H = np.zeros((n, n))
        
        # Compute second derivatives w.r.t. x1 (diagonal element H[0, 0])
        term1 = (self.g(y) * (20 - 4*self.K**2 * np.pi**2 * np.cos(2*self.K*np.pi*y1) * 2**(self.s/2)/self.K)) * (np.sin(np.pi * y1 / 2**(self.s+1) + (1 + (2**self.s - 1) / 2**(self.s+2)) * np.pi) + 1)
        term2 = 2*self.g(y) * (20*(y1 - 0.5) - 2*self.K*np.pi*np.sin(2*self.K*np.pi*y1) * 2**(self.s/2)/self.K) * (np.pi/2**(self.s+1)) * np.cos(np.pi * y1 / 2**(self.s+1) + (1 + (2**self.s - 1) / 2**(self.s+2)) * np.pi)
        term3 = -self.g(y) * self.r(y1) * (np.pi/2**(self.s+1))**2 * np.sin(np.pi * y1 / 2**(self.s+1) + (1 + (2**self.s - 1) / 2**(self.s+2)) * np.pi)
        # H[0, 0] = term1 + term2 + term3
        H[0, 0] = term1 + term2 + term3 + 0*2
        
        # Second derivatives w.r.t. x1 and xj for j > 1 (off-diagonal elements H[0, j] and H[j, 0])
        grad_g_xj = 9/(n-1)
        first_values = grad_g_xj * ((20*(y1 - 0.5) - 2*self.K*np.pi*np.sin(2*self.K*np.pi*y1) * 2**(self.s/2)/self.K) * (np.sin(np.pi * y1 / 2**(self.s+1) + (1 + (2**self.s - 1) / 2**(self.s+2)) * np.pi) + 1) + self.r(y1) * (np.pi/2**(self.s+1)) * np.cos(np.pi * y1 / 2**(self.s+1) + (1 + (2**self.s - 1) / 2**(self.s+2)) * np.pi))
        H[1:, 0] = first_values  # Set values for the first column (i > 0)
        H[0, 1:] = first_values  # Set values for the first row (j > 0)
        
        # Second derivatives w.r.t. xi and xj for i, j > 1 (H[i, j] and H[j, i])
        H[1:, 1:] = np.zeros((n-1, n-1)) #np.diag([grad_g_xj**2 * self.r(y1) * (np.sin(np.pi * y1 / 2**(self.s+1) + (1 + (2**self.s - 1) / 2**(self.s+2)) * np.pi) + 1)] * (n-1))

        # Set all diagonal elements except the first to zero
        for i in range(1, n):
            H[i, i] = 0
        
        return H
        

    def hess_f2(self, y):
        """
        The Hessian of objective function 2 
     	"""
        n = len(y)
        y1 = y[0]
        
        # Initialize Hessian matrix
        H = np.zeros((n, n))
        
        # Compute second derivatives w.r.t. x1 (diagonal element H[0, 0])
        term1 = (self.g(y) * (20 - 4*self.K**2 * np.pi**2 * np.cos(2*self.K*np.pi*y1) * 2**(self.s/2)/self.K)) * (np.cos(np.pi * y1 / 2 + np.pi) + 1)
        term2 = 2*self.g(y) * (20*(y1 - 0.5) - 2*self.K*np.pi*np.sin(2*self.K*np.pi*y1) * 2**(self.s/2)/self.K) * (-np.pi/2) * np.sin(np.pi * y1 / 2 + np.pi)
        term3 = -self.g(y) * self.r(y1) * (np.pi/2)**2 * np.cos(np.pi * y1 / 2 + np.pi)
        H[0, 0] = term1 + term2 + term3 + 0*2
        
        # Second derivatives w.r.t. x1 and xj for j > 1 (off-diagonal elements H[0, j] and H[j, 0])
        grad_g_xj = 9/(n-1)
        first_values = grad_g_xj * ((20*(y1 - 0.5) - 2*self.K*np.pi*np.sin(2*self.K*np.pi*y1) * 2**(self.s/2)/self.K) * (np.cos(np.pi * y1 / 2 + np.pi) + 1) + self.r(y1) * (-np.pi/2) * np.sin(np.pi * y1 / 2 + np.pi))
        H[1:, 0] = first_values  # Set values for the first column (i > 0)
        H[0, 1:] = first_values  # Set values for the first row (j > 0)
        
        # Second derivatives w.r.t. xi and xj for i, j > 1 (H[i, j] and H[j, i])
        H[1:, 1:] = np.zeros((n-1, n-1)) 

        # Set all diagonal elements except the first to zero
        for i in range(1, n):
            H[i, i] = 0
        
        return H
    
    
    def inequality_constraints(self, y):
        """
        Inequality constraints
     	"""
        r = 0.5 if self.tight_flag else 1
        upper_bound = y.reshape(-1,1) - np.ones((y.shape[0],1))*r
        lower_bound = -y.reshape(-1,1)  
        return np.vstack((upper_bound, lower_bound))      


    def inequality_constraints_jacob(self, y):
        """
        Jacobian of the inequality constraints
        """
        # Identity matrix for the upper bound
        I = np.eye(self.dim)
        # Combine the Jacobians for upper and lower bounds
        return np.hstack((I, -I))
    
    
    def inequality_constraints_hess(self, y):
        """
        Hessian of the inequality constraints (as a list of arrays)
        """
        num_constraints = 2 * self.dim  # Total number of constraints (upper + lower)
        # Zero Hessian for each constraint
        zero_hessian = np.zeros((self.dim, self.dim))
        hessians = [zero_hessian for _ in range(num_constraints)]
        return hessians      


    def inequality_constraints_hess_with_multipliers(self, y, lagr_mul):
        """
        Hessian of the inequality constraints (as a list of arrays)
        """
        num_constraints = 2 * self.dim  # Total number of constraints (upper + lower)
        # Zero Hessian for each constraint
        zero_hessian = np.zeros((self.dim, self.dim))
        hessians = [zero_hessian for _ in range(num_constraints)]
        return np.sum(hessians, axis=0) 
    
    
class VFM1constr:

    def __init__(self):
        
        self.dim = 2 
        self.num_obj = 3
        
        self.f_dict = {}
        self.f_dict[0] = self.f_1
        self.f_dict[1] = self.f_2
        self.f_dict[2] = self.f_3
        
        self.grad_f_dict = {}
        self.grad_f_dict[0] = self.grad_f1
        self.grad_f_dict[1] = self.grad_f2
        self.grad_f_dict[2] = self.grad_f3

        self.hess_f_dict = {}
        self.hess_f_dict[0] = self.hess_f1
        self.hess_f_dict[1] = self.hess_f2
        self.hess_f_dict[2] = self.hess_f3

        self.weight_user = {0: np.array([[0.4],[0.2],[0.4]])} 

        self.num_pareto_points = 50 

        self.constr_set = 1
        
        if self.constr_set == 0:
            self.num_ineq_constr = self.dim*2
            self.num_eq_constr = 0
        
        elif self.constr_set == 1:
            self.num_ineq_constr = 2
            self.num_eq_constr = 0   

        self.constrained = True
        

    def f_1(self, y):
        """
        Objective function 1 
    	"""
        y = np.reshape(y, (-1, 1))
        aux = np.zeros((self.dim,1))
        aux[1] = -1
        return (np.dot((y+aux).T,y+aux)).squeeze()


    def f_2(self, y):
        """
        Objective function 2 
    	"""
        y = np.reshape(y, (-1, 1))
        aux = np.zeros((self.dim,1))
        aux[1] = 1
        return (np.dot((y+aux).T,y+aux) + 1).squeeze()


    def f_3(self, y):
        """
        Objective function 3 
    	"""
        y = np.reshape(y, (-1, 1))
        aux = np.zeros((self.dim,1))
        aux[0] = -1
        return (np.dot((y+aux).T,y+aux) + 2).squeeze()
    

    def grad_f1(self, y):
        """
        The gradient of objective function 1 
    	"""
        y = np.reshape(y, (-1, 1))
        aux = np.zeros((self.dim,1))
        aux[1] = -1
        return 2*(y+aux)
    
    
    def grad_f2(self, y):
        """
        The gradient of objective function 2 
    	"""
        y = np.reshape(y, (-1, 1))
        aux = np.zeros((self.dim,1))
        aux[1] = 1
        return 2*(y+aux)


    def grad_f3(self, y):
        """
        The gradient of objective function 3
    	"""
        y = np.reshape(y, (-1, 1))
        aux = np.zeros((self.dim,1))
        aux[0] = -1
        return 2*(y+aux)
    

    def hess_f1(self, y):
        """
        The Hessian of objective function 1 
     	"""
        return 2*np.eye(self.dim)


    def hess_f2(self, y):
        """
        The Hessian of objective function 2 
     	"""
        return 2*np.eye(self.dim)


    def hess_f3(self, y):
        """
        The Hessian of objective function 3 
     	"""
        return 2*np.eye(self.dim)     
    

    def inequality_constraints(self, y):
        """
        Inequality constraints
     	"""  
        # Ensure y is a vertical 2D array
        y = y.reshape(-1, 1)

        if self.constr_set == 0:         
            # Initialize upper and lower bounds
            upper_bound = np.zeros((y.shape[0], 1))  # Placeholder for upper bounds
            lower_bound = np.zeros((y.shape[0], 1))  # Placeholder for lower bounds
    
            # Manually define each component
            upper_bound[0] = y[0] - 1
            upper_bound[1] = y[1] - 1 
            # Add more components if needed
            lower_bound[0] = -1 - y[0] 
            lower_bound[1] = -1 - y[1] 
            
            out = np.vstack((upper_bound, lower_bound)) 

        elif self.constr_set == 1:
            constr = np.zeros((2, 1))            
            constr[0] = y[0]**2 + y[1]**2 - 0.4
            constr[1] = (y[0] - 1)**2 + y[1]**2 - 1    #RHS: -1 or -0.5
            out = constr

        return out     


    def inequality_constraints_jacob(self, y):
        """
        Jacobian of the inequality constraints
        """
        if self.constr_set == 0:
            # Identity matrix for the upper bound
            I = np.eye(self.dim)
            # Combine the Jacobians for upper and lower bounds
            out = np.hstack((I, -I))

        elif self.constr_set == 1:
            jacob1 = 2*y.reshape(-1, 1)
            y = np.squeeze(y)
            jacob2 = 2*np.array([[y[0]-1],[y[1]]])
            
            out = np.hstack((jacob1, jacob2))
            
        return out
    
    
    def inequality_constraints_hess(self, y):
        """
        Hessian of the inequality constraints (as a list of arrays)
        """
        if self.constr_set == 0:
            num_constraints = 2 * self.dim  # Total number of constraints (upper + lower)
            # Zero Hessian for each constraint
            zero_hessian = np.zeros((self.dim, self.dim))
            hessians = [zero_hessian for _ in range(num_constraints)]

        elif self.constr_set == 1:
            num_constraints = 2
            # Zero Hessian for each constraint
            hessian = 2*np.eye(self.dim)
            hessians = [hessian for _ in range(num_constraints)]            

        return hessians      


    def inequality_constraints_hess_with_multipliers(self, y, lagr_mul):
        """
        Hessian of the inequality constraints (as a list of arrays)
        """
        if self.constr_set == 0:
            num_constraints = 2 * self.dim  # Total number of constraints (upper + lower)
            # Zero Hessian for each constraint
            zero_hessian = np.zeros((self.dim, self.dim))
            hessians = [zero_hessian for _ in range(num_constraints)]

        elif self.constr_set == 1:
            num_constraints = 2
            # Zero Hessian for each constraint
            hessian = np.eye(self.dim)
            hessians = [lagr_mul[i] *hessian for i in range(num_constraints)]

        return np.sum(hessians, axis=0)     


    def equality_constraints(self, y):
        """
        Equality constraints
     	"""
        return 4*y[0] - 2*y[1]**2   


    def equality_constraints_jacob(self, y):
        """
        Jacobian of the equality constraints
        """
        # Each row corresponds to the gradient of a constraint
        jacobian = np.array([
            [4, -4*y[1].item()]           
        ])
        return jacobian.T
    
    
    def equality_constraints_hess(self, y):
        """
        Hessian of the equality constraints
        """
        hessians = []

        # Hessian for g2:
        hess = np.zeros((self.dim, self.dim))
        hess[1, 1] = -4  
        hessians.append(hess)

        return hessians    


    def equality_constraints_hess_with_multipliers(self, y, lagr_mul):
        """
        Hessian of the equality constraints
        """
        hessians = []

        # Hessian for g2:
        hess = np.zeros((self.dim, self.dim))
        hess[1, 1] = -4  
        hessians.append(hess*lagr_mul[0])

        return np.sum(hessians, axis=0)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



























 

               
    
    