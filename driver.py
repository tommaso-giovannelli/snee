import numpy as np
import matplotlib.pyplot as plt
import functions as func
import snee as pn
import pickle
import json
from datetime import datetime


def run_experiment(exp_param_dict):
    """
    Auxiliary function to run the experiments.
    
    Args:
        exp_param_dict (dictionary): Dictionary with some of the attributes of the SyntheticProblem class in functions.py as keys.     
    """

    prob = func.SyntheticMultiobjProblem(name_prob_to_run=exp_param_dict['name_prob_to_run_val'])

    run = pn.Snee(prob, \
                neighborhood_type=exp_param_dict['neighborhood_type'], \
                compute_knee_solutions_flag=exp_param_dict['compute_knee_solutions_flag'], \
                algo_snee=exp_param_dict['algo_snee'], \
                plot_obj_funct_metric=exp_param_dict['plot_obj_funct_metric'], \
                plot_neighborhood=exp_param_dict['plot_neighborhood'], \
                iprint=exp_param_dict['iprint']
                )

    run_out = run.main_snee()    
    
    return run, run_out, prob


# Create a dictionary with parameters for each experiment
exp_param_dict = {}

# Create a dictionary collecting the output for each experiment
exp_out_dict = {}


#--------------------------------------------------------------------#
#------------------ Define the numerical experiments ----------------#
#--------------------------------------------------------------------#

iprint = 1 # Sets the verbosity level for printing information (higher values correspond to more detailed output)

exp_counter = 0 # Used as a counter for experiments

# ##############################
# ## Section 3 from the paper ##
# ##############################

# compute_knee_solutions_flag_val = False
# algo_snee_item = None
            
# for name_prob_to_run_val in ["ZLT1", "GRV1", "VFM1", "ZLT1q"]:
#     for neighborhood_type in [0,1,2]:
#             exp_param_dict[exp_counter] = {'name_prob_to_run_val': name_prob_to_run_val, \
#                                  'neighborhood_type': neighborhood_type, \
#                                  'compute_knee_solutions_flag': compute_knee_solutions_flag_val, \
#                                  'algo_snee': algo_snee_item, \
#                                  'plot_obj_funct_metric': True, \
#                                  'plot_neighborhood': True,\
#                                  'iprint': iprint, \
#                                  'save_output': False
#                                  }
#             exp_counter += 1

# ##############################
# ## Section 4 from the paper ##
# ##############################

# compute_knee_solutions_flag_val = True
# neighborhood_type = 1

# for name_prob_to_run_val in ["ZLT1", "GRV1", "VFM1", "GRV2"]:
#     for algo_snee_item in ["NM", "DIRECT"]:
#             exp_param_dict[exp_counter] = {'name_prob_to_run_val': name_prob_to_run_val, \
#                                  'neighborhood_type': neighborhood_type, \
#                                  'compute_knee_solutions_flag': compute_knee_solutions_flag_val, \
#                                  'algo_snee': algo_snee_item, \
#                                  'plot_obj_funct_metric': True, \
#                                  'plot_neighborhood': True,\
#                                  'iprint': iprint, \
#                                  'save_output': False
#                                  }
#             exp_counter += 1

# ##############################
# ## Section 5 from the paper ##
# ##############################

# compute_knee_solutions_flag_val = True
# neighborhood_type = 1

# for name_prob_to_run_val in ["DAS1", "DO2DK", "DO2DKtight", "VFM1constr"]:
#     for algo_snee_item in ["NM", "DIRECT"]:
#             exp_param_dict[exp_counter] = {'name_prob_to_run_val': name_prob_to_run_val, \
#                                  'neighborhood_type': neighborhood_type, \
#                                  'compute_knee_solutions_flag': compute_knee_solutions_flag_val, \
#                                  'algo_snee': algo_snee_item, \
#                                  'plot_obj_funct_metric': True, \
#                                  'plot_neighborhood': True,\
#                                  'iprint': iprint, \
#                                  'save_output': False
#                                  }
#             exp_counter += 1            

##################################
## Custom Numerical Experiments ##
##################################

exp_param_dict[exp_counter] = {'name_prob_to_run_val': "DO2DK", \
                     'neighborhood_type': 1, \
                     'compute_knee_solutions_flag': True, \
                     'algo_snee': "NM", \
                     'plot_obj_funct_metric': True, \
                     'plot_neighborhood': True,\
                     'iprint': iprint, \
                     'save_output': False
                     }

    
#--------------------------------------------------------------------#
#-------------- Run the experiments and make the plots --------------#
#--------------------------------------------------------------------#

for exp_to_run in range(len(exp_param_dict)):

        print('\n******************************************************')
        print('\nExperiment: ',exp_to_run,'/',len(exp_param_dict),' - Problem: ',exp_param_dict[exp_to_run]['name_prob_to_run_val'], \
              ' - Knee Solutions? ',exp_param_dict[exp_to_run]['compute_knee_solutions_flag'],' - Algorithm: ',exp_param_dict[exp_to_run]['algo_snee'])  
        print('******************************************************\n')
        
        run, run_out, prob = run_experiment(exp_param_dict[exp_to_run])
    
        exp_out_dict[exp_to_run] = {'run': run, \
                       'f_pareto_value_list_dict': run_out[0], \
                       'vars_pareto_list': run_out[1], \
                       'simplex_set_discr': run_out[2], \
                       'f_dict_user_per_weight': run_out[3], \
                       'y_user_per_weight': run_out[4], \
                       'weight_user': run_out[5], \
                       'f_ellipsoid_list_dict': run_out[6], \
                       'y_ellipsoid_list': run_out[7], \
                       'simplex_set_discr_ellipsoid': run_out[8], \
                       'A': run_out[9], \
                       'B': run_out[10], \
                       'rhs_ellipsoid': run_out[11], \
                       'weight_iterates_list': run_out[12], \
                       'f_value_iterates_list': run_out[13], \
                       'y_iterates_list': run_out[14], \
                       'obj_funct_values_list': run_out[15], \
                       'metric_values_list': run_out[16], \
                       'value_test_obj_list': run_out[17], \
                       'exp_param_dict_elem': exp_param_dict[exp_to_run]}
                        
        # Save exp_out_dict[exp_to_run]    
        if exp_param_dict[exp_to_run]['save_output']: 
            string_dict = r'OutputResults/' + exp_param_dict[exp_to_run]['name_prob_to_run_val'] + '_dict.pkl'
            with open(string_dict, 'wb') as f:
                pickle.dump(exp_out_dict[exp_to_run], f)
        
                
        if 2 <= prob.prob.num_obj <= 3:
            
            ########################
            # # Plot the simplex set
            ########################
            
            simplex_set_discr = exp_out_dict[exp_to_run]['simplex_set_discr']
            weight_user = exp_out_dict[exp_to_run]['weight_user']
            simplex_set_discr_ellipsoid = exp_out_dict[exp_to_run]['simplex_set_discr_ellipsoid'] 
            rhs_ellipsoid = exp_out_dict[exp_to_run]['rhs_ellipsoid'] 
            weight_iterates_list = exp_out_dict[exp_to_run]['weight_iterates_list']
            
            fig = plt.figure()
            
            if prob.prob.num_obj == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(simplex_set_discr[0], simplex_set_discr[1], simplex_set_discr[2], color='gray', s=5)
                if exp_param_dict[exp_to_run]['plot_neighborhood']:
                    ax.scatter(simplex_set_discr_ellipsoid[0], simplex_set_discr_ellipsoid[1], simplex_set_discr_ellipsoid[2], color='lightcoral', s=10) #color='red', s=10) #10
                adj=0.1
                for i in range(len(weight_user)):
                    ax.scatter(weight_user[i][0]+adj, weight_user[i][1]+adj, weight_user[i][2]+adj, color='black', s=50)
                # Knee solutions
                if exp_param_dict[exp_to_run]['compute_knee_solutions_flag']:
                    print('Starting point: ', weight_iterates_list[0][0], weight_iterates_list[0][1], weight_iterates_list[0][2])
                    if exp_param_dict[exp_to_run]['algo_snee'] == "NM":
                        ax.scatter(weight_iterates_list[0][0]+adj, weight_iterates_list[0][1]+adj, weight_iterates_list[0][2]+adj, color='cyan', s=100, marker='*')
                    elif exp_param_dict[exp_to_run]['algo_snee'] == "DIRECT":
                        ax.scatter(weight_iterates_list[0][0]+adj, weight_iterates_list[0][1]+adj, weight_iterates_list[0][2]+adj, color='purple', s=25, marker='*')
                    for i in range(1, len(weight_iterates_list) - 1):
                        ax.scatter(weight_iterates_list[i][0]+adj, weight_iterates_list[i][1]+adj, weight_iterates_list[i][2]+adj, color='purple', s=25, marker='*')
                
                # Set labels for the axes
                ax.set_xlabel("$\lambda_1$", fontsize = 20)
                ax.set_ylabel("$\lambda_2$", fontsize = 20)
                ax.set_zlabel("$\lambda_3$", fontsize = 20)
                
            elif prob.prob.num_obj == 2:
                ax = fig.add_subplot(111)
                ax.scatter(simplex_set_discr[0], simplex_set_discr[1], color='gray', s=5)
                if exp_param_dict[exp_to_run]['plot_neighborhood']:
                    ax.scatter(simplex_set_discr_ellipsoid[0], simplex_set_discr_ellipsoid[1], color='lightcoral', s=10)
                for i in range(len(weight_user)):
                    ax.scatter(weight_user[i][0], weight_user[i][1], color='black', s=60)
                # Knee solutions
                if exp_param_dict[exp_to_run]['compute_knee_solutions_flag']:
                    print('Starting point: ', weight_iterates_list[0][0], weight_iterates_list[0][1])
                    if exp_param_dict[exp_to_run]['algo_snee'] == "NM":
                        ax.scatter(weight_iterates_list[0][0], weight_iterates_list[0][1], color='cyan', s=100, marker='*')
                    elif exp_param_dict[exp_to_run]['algo_snee'] == "DIRECT":
                        ax.scatter(weight_iterates_list[0][0], weight_iterates_list[0][1], color='purple', s=25, marker='*')
                    for i in range(1, len(weight_iterates_list) - 1):
                        ax.scatter(weight_iterates_list[i][0], weight_iterates_list[i][1], color='purple', s=25, marker='*')
                
                # Set labels for the axes
                ax.set_xlabel("$\lambda_1$", fontsize = 20)
                ax.set_ylabel("$\lambda_2$", fontsize = 20)
            
                
            # Matrix for ellipsoid
            A = exp_out_dict[exp_to_run]['A']
                
            
            plt.title("Parameter Space")
            
            # plt.legend()
            plt.tight_layout()
            
            fig = plt.gcf()
            
            # # Uncomment the next line to save the plot
            # string = 'fig_simplex_set.pdf'
            # fig.savefig(string)
            
            plt.show()
            
            
        if 2 <= prob.prob.num_obj <= 3:    
            ##########################
            # # Plot the Pareto fronts
            ##########################
            
            f_value_list_dict = exp_out_dict[exp_to_run]['f_pareto_value_list_dict']
            f_value_dict_user_per_weight = exp_out_dict[exp_to_run]['f_dict_user_per_weight']
            f_value_ellipsoid_list_dict = exp_out_dict[exp_to_run]['f_ellipsoid_list_dict']
            rhs_ellipsoid = exp_out_dict[exp_to_run]['rhs_ellipsoid']
            f_value_iterates_list = exp_out_dict[exp_to_run]['f_value_iterates_list']
            
            fig = plt.figure()
        
            if prob.prob.num_obj == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(f_value_list_dict[0], f_value_list_dict[1], f_value_list_dict[2], color='#2ca02c', s=4) #color='#2ca02c', s=4)
                if exp_param_dict[exp_to_run]['plot_neighborhood']:
                    ax.scatter(f_value_ellipsoid_list_dict[0], f_value_ellipsoid_list_dict[1], f_value_ellipsoid_list_dict[2], color='lightcoral', s=4, marker='D') #color='red', s=4, marker='D') #1
                # ax.scatter(f_value_ellipsoid_list_dict[0], f_value_ellipsoid_list_dict[1], f_value_ellipsoid_list_dict[2], color='#2ca02c', s=4)
                adj = 0#-0.15
                for i in range(len(f_value_dict_user_per_weight)):
                    ax.scatter(f_value_dict_user_per_weight[i][0]+adj-0.15, f_value_dict_user_per_weight[i][1]+adj-0.15, f_value_dict_user_per_weight[i][2]+adj, color='black', s=50) 
                # Knee solution
                if exp_param_dict[exp_to_run]['compute_knee_solutions_flag']:
                    if exp_param_dict[exp_to_run]['algo_snee'] == "NM":
                        ax.scatter(f_value_iterates_list[0][0]+adj-0.15, f_value_iterates_list[0][1]+adj, f_value_iterates_list[0][2]+adj, color='cyan', s=100, marker='*')
                    elif exp_param_dict[exp_to_run]['algo_snee'] == "DIRECT":
                        ax.scatter(f_value_iterates_list[0][0]+adj-0.15, f_value_iterates_list[0][1]+adj, f_value_iterates_list[0][2]+adj, color='purple', s=25, marker='*')
                    for i in range(1,len(f_value_iterates_list)-1):
                        ax.scatter(f_value_iterates_list[i][0]+adj-0.15, f_value_iterates_list[i][1]+adj, f_value_iterates_list[i][2]+adj, color='purple', s=25, marker='*') 
        
                # Get x-axis and y-axis limits
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                z_min, z_max = ax.get_zlim()
                
                print(f"Objective x-axis limits: min={x_min}, max={x_max}")
                print(f"Objective y-axis limits: min={y_min}, max={y_max}")
                print(f"Objective z-axis limits: min={z_min}, max={z_max}")
                
                # Set labels for the axes
                ax.set_xlabel("$f_1$", fontsize = 20)
                ax.set_ylabel("$f_2$", fontsize = 20)
                ax.set_zlabel("$f_3$", fontsize = 20)
        
                if prob.prob.__class__.__name__ == 'VFM1constr':
                    ax.set_xlim(0.024999624644386, 2.7750012005654)
                    ax.set_ylim(1.0249996246443858, 3.7750012005654)
                    ax.set_zlim(2.0720220394261135, 3.0630674994418063)
        
                if prob.prob.__class__.__name__ == "ZLT1" or prob.prob.__class__.__name__ == "VFM1":
                    # Reduce the font size of the tick labels for f1 and f2 axes
                    ax.tick_params(axis='x', which='major', labelsize=6)
                    ax.tick_params(axis='y', which='major', labelsize=6)
        
                # Set equal aspect ratio
                ax.set_box_aspect([1, 1, 1])  
        
                # Calculate the data range for each axis
                x_range = ax.get_xlim()
                y_range = ax.get_ylim()
                z_range = ax.get_zlim()
                
                # Find the max range across all axes
                max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]) / 2.0
                
                # Calculate midpoints for all axes
                x_mid = (x_range[1] + x_range[0]) / 2.0
                y_mid = (y_range[1] + y_range[0]) / 2.0
                z_mid = (z_range[1] + z_range[0]) / 2.0
                
                # Set limits for each axis to make the plot have equal aspect ratios
                ax.set_xlim(x_mid - max_range, x_mid + max_range)
                ax.set_ylim(y_mid - max_range, y_mid + max_range)
                ax.set_zlim(z_mid - max_range, z_mid + max_range)
        
        
            elif prob.prob.num_obj == 2:
                ax = fig.add_subplot(111)
                ax.scatter(f_value_list_dict[0], f_value_list_dict[1], color='#2ca02c', s=5)
                if exp_param_dict[exp_to_run]['plot_neighborhood']:
                    ax.scatter(f_value_ellipsoid_list_dict[0], f_value_ellipsoid_list_dict[1], color='lightcoral', s=5, marker='D') 
                for i in range(len(f_value_dict_user_per_weight)):
                    ax.scatter(f_value_dict_user_per_weight[i][0], f_value_dict_user_per_weight[i][1], color='black', s=60) 
                
                # Knee solutions
                if exp_param_dict[exp_to_run]['compute_knee_solutions_flag']:
                    if exp_param_dict[exp_to_run]['algo_snee'] == "NM":
                        ax.scatter(f_value_iterates_list[0][0], f_value_iterates_list[0][1], color='cyan', s=100, marker='*')
                    elif exp_param_dict[exp_to_run]['algo_snee'] == "DIRECT":
                        ax.scatter(f_value_iterates_list[0][0], f_value_iterates_list[0][1], color='purple', s=25, marker='*')
                    for i in range(1,len(f_value_iterates_list)-1):
                        ax.scatter(f_value_iterates_list[i][0], f_value_iterates_list[i][1], color='purple', s=25, marker='*') 
        
        
                # Get x-axis and y-axis limits
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                
                print(f"Objective x-axis limits: min={x_min}, max={x_max}")
                print(f"Objective y-axis limits: min={y_min}, max={y_max}")
                
                # Set labels for the axes
                ax.set_xlabel("$f_1$", fontsize = 20)
                ax.set_ylabel("$f_2$", fontsize = 20)
        
                if prob.prob.__class__.__name__ == 'DO2DK':
                    ax.set_xlim(-0.4468979667231137, 9.384857301185388)
                    ax.set_ylim(-0.4250000000018059, 8.925000000037922)
        
                # Set equal scaling
                plt.gca().set_aspect('equal', adjustable='box')
            
            plt.title("Objective Space")
            
            # plt.legend()
            plt.tight_layout()
            
            fig = plt.gcf()
            
            # # Uncomment the next line to save the plot
            # string = 'fig_solution_pareto_fronts.pdf'
            # fig.savefig(string)
            
            plt.show()
            
            
        if 2 <= prob.prob.num_obj:    
            ##########################
            # # Pareto Front Metrics
            ##########################  
        
            f_value_list_dict = exp_out_dict[exp_to_run]['f_pareto_value_list_dict']
            f_value_dict_user_per_weight = exp_out_dict[exp_to_run]['f_dict_user_per_weight']
            f_value_ellipsoid_list_dict = exp_out_dict[exp_to_run]['f_ellipsoid_list_dict']
            rhs_ellipsoid = exp_out_dict[exp_to_run]['rhs_ellipsoid']
            obj_funct_values_list = exp_out_dict[exp_to_run]['obj_funct_values_list']
            metric_values_list = exp_out_dict[exp_to_run]['metric_values_list']
            value_test_obj_list = exp_out_dict[exp_to_run]['value_test_obj_list']
            
            if exp_param_dict[exp_to_run]['compute_knee_solutions_flag'] and exp_param_dict[exp_to_run]['save_output']:
                # WRITE FILES
                # Get current date and time
                now = datetime.now()
                # Format it as a string for the filename
                date_str = now.strftime("%Y%m%d_%H%M%S")  
                
                # Create the filename with the date and time
                filename_obj = f"OutputResults/{prob.prob.__class__.__name__}_obj_funct_values_list_{date_str}.json"
                filename_metric = f"OutputResults/{prob.prob.__class__.__name__}_metric_values_list_{date_str}.json"
        
                # Save the list to a JSON file
                with open(filename_obj, 'w') as file:
                    json.dump(obj_funct_values_list, file) 
                
                # Save the list to a JSON file
                with open(filename_metric, 'w') as file:
                    json.dump(metric_values_list, file)    
        
                
            if exp_param_dict[exp_to_run]['compute_knee_solutions_flag'] and exp_param_dict[exp_to_run]['plot_obj_funct_metric']: # Plot two windows, one inside the other
                # Generate x-axis values (optional, can be indices of the list)
                x_values = range(len(obj_funct_values_list))
                
                # Convert list to numpy array for easier manipulation
                arr = np.array(obj_funct_values_list)
                
                # Create a figure and axis
                fig, ax = plt.subplots(figsize=(10, 5))  
                
                # Plot the main plot (objective function values)
                ax.plot(x_values, arr, marker='o', linestyle='-', color='b', label='MCF', markersize=2)
                
                # Add labels and title
                ax.set_xlabel('Iterations')
                
                # Add grid
                ax.grid(False)
                
                # Show legend
                ax.legend(loc='upper right')
                # ax.legend()
                
                # Add inset plot manually
                # Specify the position and size of the inset plot in normalized figure coordinates (0 to 1)
                # Example: (left, bottom, width, height) in figure fraction coordinates
                if exp_param_dict[exp_to_run]['algo_snee'] == "NM":
                    inset_ax = fig.add_axes([0.52, 0.25, 0.35, 0.35])  # Adjust values as needed
                elif exp_param_dict[exp_to_run]['algo_snee'] == "DIRECT":
                    inset_ax = fig.add_axes([0.52, 0.25, 0.35, 0.35])  # Adjust values as needed
            
                # Plot the smaller plot (metric values)
                inset_ax.plot(x_values, metric_values_list, marker='s', linestyle='-', color='r', label='MCM', markersize=2)
                
                # Set labels and title for the inset plot if needed
                inset_ax.set_xlabel('Iterations')
                # inset_ax.set_title('OS Metric')
                
                # Add grid to the inset plot
                inset_ax.grid(False)
                
                # Show legend for the inset plot
                # inset_ax.legend(loc='lower right')
                inset_ax.legend(loc='upper right')
                
                # Show plot
                plt.show()    
            
            
        if 2 <= prob.prob.dim <= 3:
                ##########################
                # # Plot the Pareto points
                ##########################
                
                # plt.figure()
                
                y_list = exp_out_dict[exp_to_run]['vars_pareto_list']
                y_user_per_weight = exp_out_dict[exp_to_run]['y_user_per_weight']
                y_ellipsoid_list = exp_out_dict[exp_to_run]['y_ellipsoid_list']
                rhs_ellipsoid = exp_out_dict[exp_to_run]['rhs_ellipsoid']
                y_iterates_list = exp_out_dict[exp_to_run]['y_iterates_list']
                
                fig = plt.figure()
            
                if prob.prob.dim == 3:
            
                    y_list_1 = [arr[0, 0] for arr in y_list]
                    y_list_2 = [arr[1, 0] for arr in y_list]
                    y_list_3 = [arr[2, 0] for arr in y_list]
                    
                    y_ellipsoid_list_1 = [arr[0, 0] for arr in y_ellipsoid_list]
                    y_ellipsoid_list_2 = [arr[1, 0] for arr in y_ellipsoid_list]
                    y_ellipsoid_list_3 = [arr[2, 0] for arr in y_ellipsoid_list]
                    
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(y_list_1, y_list_2, y_list_3, color='lightblue', s=5) #color='#1f77b4', s=5)
                    if exp_param_dict[exp_to_run]['plot_neighborhood']:
                        ax.scatter(y_ellipsoid_list_1, y_ellipsoid_list_2, y_ellipsoid_list_3, color='lightcoral', s=5, marker='D') #color='red', s=5, marker='D') #10
                    adj = 0.1
                    for i in range(len(y_user_per_weight)):
                        ax.scatter(y_user_per_weight[i][0]+adj, y_user_per_weight[i][1]+adj, y_user_per_weight[i][2]+adj, color='black', s=50)
                    # Knee solutions
                    if exp_param_dict[exp_to_run]['compute_knee_solutions_flag']:
                        if exp_param_dict[exp_to_run]['algo_snee'] == "NM":
                            ax.scatter(y_iterates_list[0][0]+adj, y_iterates_list[0][1]+adj, y_iterates_list[0][2]+adj, color='cyan', s=100, marker='*')
                        elif exp_param_dict[exp_to_run]['algo_snee'] == "DIRECT":
                            ax.scatter(y_iterates_list[0][0]+adj, y_iterates_list[0][1]+adj, y_iterates_list[0][2]+adj, color='purple', s=25, marker='*')
                        for i in range(1,len(y_iterates_list)-1):
                            ax.scatter(y_iterates_list[i][0]+adj, y_iterates_list[i][1]+adj, y_iterates_list[i][2]+adj, color='purple', s=25, marker='*')            
        
                    # Get x-axis and y-axis limits
                    x_min, x_max = ax.get_xlim()
                    y_min, y_max = ax.get_ylim()
                    z_min, z_max = ax.get_zlim()
                    
                    print(f"Decision x-axis limits: min={x_min}, max={x_max}")
                    print(f"Decision y-axis limits: min={y_min}, max={y_max}")
                    print(f"Decision z-axis limits: min={z_min}, max={z_max}")
                    
                    # Set labels for the axes
                    ax.set_xlabel("$x_1$", fontsize = 20)
                    ax.set_ylabel("$x_2$", fontsize = 20)
                    ax.set_zlabel("$x_3$", fontsize = 20)
            
                elif prob.prob.dim == 2:
                    
                    y_list_1 = [arr[0, 0] for arr in y_list]
                    y_list_2 = [arr[1, 0] for arr in y_list]
                    
                    y_ellipsoid_list_1 = [arr[0, 0] for arr in y_ellipsoid_list]
                    y_ellipsoid_list_2 = [arr[1, 0] for arr in y_ellipsoid_list]
                               
                    
                    ax = fig.add_subplot(111)
                    ax.scatter(y_list_1, y_list_2, color='lightblue', s=5) #color='#1f77b4', s=5)
                    if exp_param_dict[exp_to_run]['plot_neighborhood']:
                        ax.scatter(y_ellipsoid_list_1, y_ellipsoid_list_2, color='lightcoral', s=5, marker='D') #color='red', s=5, marker='D') 
                    for i in range(len(y_user_per_weight)):
                        ax.scatter(y_user_per_weight[i][0], y_user_per_weight[i][1], color='black', s=60)
                    # Knee solutions
                    if exp_param_dict[exp_to_run]['compute_knee_solutions_flag']:
                        if exp_param_dict[exp_to_run]['algo_snee'] == "NM":
                            ax.scatter(y_iterates_list[0][0], y_iterates_list[0][1], color='cyan', s=100, marker='*')
                        elif exp_param_dict[exp_to_run]['algo_snee'] == "DIRECT":
                            ax.scatter(y_iterates_list[0][0], y_iterates_list[0][1], color='purple', s=25, marker='*')
                        for i in range(1,len(y_iterates_list)-1):
                            ax.scatter(y_iterates_list[i][0], y_iterates_list[i][1], color='purple', s=25, marker='*')            
        
                    # Get x-axis and y-axis limits
                    x_min, x_max = ax.get_xlim()
                    y_min, y_max = ax.get_ylim()
                    
                    print(f"Decision x-axis limits: min={x_min}, max={x_max}")
                    print(f"Decision y-axis limits: min={y_min}, max={y_max}")
                    
                    # Set labels for the axes
                    ax.set_xlabel("$x_1$", fontsize = 20)
                    ax.set_ylabel("$x_2$", fontsize = 20) 
                    
                    if prob.prob.__class__.__name__ == 'VFM1constr':
                        ax.set_xlim(-0.03017475325677394, 0.6640100315583426)
                        ax.set_ylim(-0.6600003782210433, 0.6600003782210432)
                
                
                # Matrix for ellipsoid
                B = exp_out_dict[exp_to_run]['B']
                
                
                # Contour plot GRV1
                if prob.prob.__class__.__name__ == 'GRV1' and False:
                    x_vals = np.arange(-1, 2,0.05) 
                    y_vals = np.arange(-1.5,1.5,0.05)
                    X, Y = np.meshgrid(x_vals, y_vals)
                    weight_dummy = weight_user #{0: np.array([[1],[0],[0]])} #simplex_set_discr_ellipsoid #weight_user
                    Z = weight_dummy[0][0]*(prob.prob.a1*X + prob.prob.a2*Y + X*prob.prob.H3*Y + 0.5*Y*prob.prob.H2*Y + 0.5*X*prob.prob.H1*X)+\
                        weight_dummy[0][1]*(prob.prob.a3*X + prob.prob.a4*Y + X*prob.prob.H6*Y + 0.5*Y*prob.prob.H5*Y + 0.5*X*prob.prob.H4*X)+\
                        weight_dummy[0][2]*(prob.prob.a5*X + prob.prob.a6*Y + X*prob.prob.H9*Y + 0.5*Y*prob.prob.H8*Y + 0.5*X*prob.prob.H7*X)
                    cp = plt.contour(X, Y, Z, levels=100, linewidths=0.5)
        
                
                plt.title("Decision Space")
                
                # plt.legend(frameon=True, fontsize = 12)
                plt.tight_layout()
                
                fig = plt.gcf()
                
                # # Uncomment the next line to save the plot
                # string = 'fig_solution_2D.pdf'
                # fig.savefig(string)
                
                plt.show()
        
    














