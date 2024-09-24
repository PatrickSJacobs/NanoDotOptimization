
from jmetal.core.problem import OnTheFlyFloatProblem
from jmetal.algorithm.multiobjective.gde3 import GDE3
#from jmetal.util.evaluator import MultiprocessEvaluator
#from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.comparator import DominanceComparator
#from jmetal.util.solution import get_non_dominated_solutions
#from jmetal.algorithm.multiobjective.nsgaii import NSGAII
#from jmetal.operator import PolynomialMutation, SBXCrossover
#from jmetal.problem.multiobjective.zdt import ZDT1Modified
from jmetal.util.evaluator import MultiprocessEvaluator
#from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from datetime import datetime
import csv
import numpy as np
import pandas as pd
import os
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

current_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")# Getting the current time
main_home_dir = "/home1/08809/tg881088/" # Home directory for optimization
folder_name = "opt_%s" % str(current_time)# Folder name for optimization files
file_home_path = main_home_dir + folder_name + "_processed/" # Folder name for optimization files
main_work_dir = "/work2/08809/tg881088/" # Home directory for optimization
file_work_path = main_work_dir + folder_name + "_raw/" # Folder name for optimization files
progress_file = file_home_path + "progress.txt"
os.mkdir(file_home_path)# Making folder name for optimization files
os.mkdir(file_work_path)# Making folder name for data log
file_naught = open(progress_file, 'w')
file_naught.writelines(["Beginning optimization %s" % "\n"])
file_naught.close()

#execution_dictionary = {}

def printing(string):

    file_printout = open(progress_file, 'r').readlines()
    lines = file_printout + [f"{str(string)}\n"]
    file_printer = open(progress_file, 'w')
    file_printer.writelines(lines)
    file_printer.close()
    print(string)
    pass

pretraining_data_path = os.path.join(main_work_dir, 'ag-dot-angle-pretraining-unpruned.csv')  # Replace with your CSV file path
data = pd.read_csv(pretraining_data_path)

# Inputs
X = data[['sr', 'ht', 'cs', 'theta_deg']].values

y_c = data['c-param'].values  # Objective 1
y_b = data['b-param'].values  # Objective 2
y_bvar = data['b_var'].values  # Objective 3

# Train/test split
X_train, X_test, y_c_train, y_c_test = train_test_split(X, y_c, test_size=0.1, random_state=42)
_, _, y_b_train, y_b_test = train_test_split(X, y_b, test_size=0.1, random_state=42)
_, _, y_bvar_train, y_bvar_test = train_test_split(X, y_bvar, test_size=0.1, random_state=42)

del _

# Train XGBoost models for each objective
model_c = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model_b = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model_bvar = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

# Train the models
model_c.fit(X_train, y_c_train)
model_b.fit(X_train, y_b_train)
model_bvar.fit(X_train, y_bvar_train)

del X_train
del y_c_train
del X_test
del y_c_test
del y_b_train
del y_bvar_train
del y_bvar_test

def b(x): return model_b.predict(np.array(x).reshape(1, -1))
def c(x): return model_c.predict(np.array(x).reshape(1, -1))
def b_var(x): return model_bvar.predict(np.array(x).reshape(1, -1))
#def c_var(x: [float]): return get_values(x, "c_var")
def c_upper_constraint(x): return 20 - c(x)
def c_lower_constraint(x): return c(x)
def b_lower_constraint(x): return b(x) - 1  # b-param should be >= 1
def b_upper_constraint(x): return 50 -  b(x)  # b-param should be <= 60
def b_var_constraint(x): return 10 - b_var(x)


#bounds = {'sr': (0.001 * 5, 0.001 * 125), 'ht': (0.001 * 50, 0.001 * 100), 'cs': (0.001 * 25, 0.001 * 250), 'theta_deg': (0.0, 0.0)}# Bounds for optimization

'''.add_variable(0.001 * 5, 0.001 * 125)
    .add_variable(0.001 * 50, 0.001 * 100)
    .add_variable(0.001 * 25, 0.001 * 250)
    #.add_variable(0.001 * 250, 0.001 * 250)
    #.add_variable(0.0, 0.0)
    .add_variable(0.0, 0.0)'''
    
problem = (
    OnTheFlyFloatProblem()
    .set_name("Testing")
    .add_variable(0.001 * 5, 0.001 * 125)
    .add_variable(0.001 * 50, 0.001 * 100)
    .add_variable(0.001 * 25, 0.001 * 250)
    #.add_variable(0.001 * 250, 0.001 * 250)
    #.add_variable(0.0, 0.0)
    .add_variable(0.0, 0.0)
    .add_function(c)
    .add_function(b)
    .add_function(b_var)
    #.add_function(c_var)
    .add_constraint(b_lower_constraint)
    .add_constraint(b_upper_constraint)
    .add_constraint(c_lower_constraint)
    .add_constraint(c_upper_constraint)
    .add_constraint(b_var_constraint)
)

if __name__ == "__main__":

    with open(file_home_path + "calc_log_obj.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "sr", "ht", "cs", "theta_deg", "b-param", "c-param", "b_var", "c_var","execution time", "step count"])
        file.close()

    max_evaluations = 640
    #max_evaluations = 8

    '''

    algorithm = NSGAII(
        population_evaluator=MultiprocessEvaluator(processes=16),
        problem=problem,
        population_size=16,
        offspring_population_size=16,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        #dominance_comparator=DominanceComparator(),
    )

    '''
    
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from sklearn.cluster import KMeans

    def select_diverse_solutions(pareto_parameters, pareto_objectives, m):
        # Combine parameters and objectives for clustering
        combined = np.hstack((pareto_parameters, pareto_objectives))
        
        # Normalize the data
        combined_normalized = (combined - np.min(combined, axis=0)) / (np.max(combined, axis=0) - np.min(combined, axis=0))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=m, random_state=42)
        kmeans.fit(combined_normalized)
        
        # Select the solution closest to each cluster center
        selected_indices = []
        for i in range(m):
            cluster_points = combined_normalized[kmeans.labels_ == i]
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            closest_point_index = np.argmin(distances)
            selected_indices.append(np.where((combined_normalized == cluster_points[closest_point_index]).all(axis=1))[0][0])
        
        return selected_indices

    #df1 = pd.read_csv(main_work_dir + "ag-dot-angle-pretraining.csv")
    df1 = pd.read_csv(main_work_dir + "ag-dot-angle-pretraining.csv")

    parameters = df1[['sr', 'ht', 'cs', 'theta_deg']].values
    objectives = df1[['c-param', 'b-param', 'b_var']].values

    # Find the Pareto front
    nds = NonDominatedSorting()
    pareto_front_indices = nds.do(objectives, only_non_dominated_front=True)

    # Extract Pareto front solutions
    pareto_parameters = parameters[pareto_front_indices]
    pareto_objectives = objectives[pareto_front_indices]

    # Select m diverse solutions
    population_size = 32
    selected_indices = select_diverse_solutions(pareto_parameters, pareto_objectives, population_size)

    # Extract the selected solutions
    selected_parameters = pareto_parameters[selected_indices]
    selected_objectives = pareto_objectives[selected_indices]

    # Output the selected solutions
    print(f"Number of selected Pareto-optimal solutions: {population_size}")
    for i in range(population_size):
        print(f"\nSolution {i+1}:")
        print(f"Parameters: {selected_parameters[i]}")
        print(f"Objectives: {selected_objectives[i]}")

    # Prepare the selected solutions for GDE3
    gde3_initial_population = selected_parameters

    print("\nInitial population for GDE3:")
    print(gde3_initial_population)

    #sys.exit()
    
    algorithm = GDE3(
        population_evaluator=MultiprocessEvaluator(processes=16),
        problem=problem,
        #population_size=16,
        population_size=population_size,
        cr=0.9,
        f=0.4,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        dominance_comparator=DominanceComparator(),
    )
    
    algorithm.solutions = gde3_initial_population

    algorithm.run()
    front = algorithm.result()
    print(front)

    for sol in range(len(front)):
        vars = front[sol].variables
        print(f'(Solution #{sol + 1}): Variables={front[sol].variables}; Objectives={front[sol].objectives}')
        printing(f'(Solution #{sol + 1}):')
        printing(f'             Variables={vars}')
        printing(f'             Objectives={front[sol].objectives}')

    printing(f"Computing time: {algorithm.total_computing_time}")


