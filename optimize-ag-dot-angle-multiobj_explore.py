
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
import time
import string
import random
from time import sleep
import csv
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import statistics
from scipy.signal import find_peaks
import sys
import traceback
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


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

def check_log(filename: str, param: str):
    df = pd.read_csv(file_home_path + "calc_log_obj.csv")
    return list(dict(df.loc[df['filename'] == filename])[param])

def make_filename(sr, ht, cs, theta_deg):
    display_theta_deg = str(round(theta_deg if theta_deg > 0 else theta_deg + 360.0,
                                  1)).replace(".", "_")  # angle to be used

    filename = "%s_sr_%s_ht_%s_cs_%s_theta_deg_%s" % (str(folder_name),
                                                      str(round(sr * 10000, 1)).replace(".", "_") + "nm",
                                                      str(round(ht * 10000, 1)).replace(".", "_") + "nm",
                                                      str(round(cs * 10000, 1)).replace(".", "_") + "nm",
                                                      display_theta_deg,
                                                      )  # filename to be used
    return filename

def obj_func_run(x: [float]):
    """
    (3) Running the Optimization with Test Values
        - Given the test parameters, construct a optimization to get the reflectance from the situation with respect to the parameters
        - Optimization is performed and the worker waits until the data is ready for extraction
        - The data is extracted, logged, and sent to obj_func_calc which performs the final processing of the wavelength and reflectance data
        - The objective function results are logged and any unnecessary files are deleted.
        - The objective function results are sent back to the optimizer
    """
    sr = x[0]
    ht = x[1]
    cs = x[2]
    theta_deg = x[3]
    #cs = 0.001 * 250
    #theta_deg = x[2]

    sleep(10)# Sleep to give code time to process parallelization

    # Parameters to be used in current evaluation
    printing((sr, ht, cs, theta_deg))

    filename = make_filename(sr, ht, cs, theta_deg)

    #Creating Scheme executable for current optimization; ag-dot-angle0.ctl cannot be used simultaneously with multiple workers)
    executable = open(main_home_dir + "NanoDotOptimization/ag-dot-angle0.ctl", 'r')
    lines = executable.readlines()
    code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    new_name = file_home_path + "ag-dot-angle" + code
    new_file = new_name + ".ctl"
    file0 = open(new_file, 'w')
    file0.writelines(lines)
    file0.close()

    # Creating ticker file to make sure the data is created and stable for processing
    ticker_file = file_home_path + "ticker" + code + ".txt"
    file2 = open(ticker_file, 'w')
    file2.write("0")
    file2.close()

    # Creation of simulation "subjob" file
    sbatch_file = file_home_path + "/" + str(filename) + ".txt"
    file1 = open(sbatch_file, 'w')

    air_file = "%sair-angle_%s" % (file_home_path, filename)
    metal_file = "%sag-dot-angle_%s" % (file_home_path, filename)
    
    air_raw_path = air_file + ".out"
    metal_raw_path = metal_file + ".out"
    air_data_path = air_file + ".dat"
    metal_data_path = metal_file + ".dat"
    # cell_size = 2*(sr + cs)
    cell_size = 2 * sr + cs
    
    info_file = new_name + ".txt"

    with open(info_file, "w") as f:
        for item in [
            progress_file, 
            air_data_path, 
            metal_data_path, 
            file_home_path, 
            file_work_path, 
            filename, 
            ticker_file, 
            air_raw_path, 
            metal_raw_path, 
            sr, 
            ht, 
            cs, 
            theta_deg
                    ]:
            f.write(f"{item}\n")

    file1.writelines(["#!/bin/bash%s" % "\n",
                      "#SBATCH -J myMPI%s" % "\n",
                      "#SBATCH -o myMPI.%s%s" % ("o%j", "\n"),
                      "#SBATCH -n 32%s" % "\n",
                      "#SBATCH -N 1%s" % "\n",
                      "#SBATCH --mail-user=pjacobs7@eagles.nccu.edu%s" % "\n",
                      "#SBATCH --mail-type=all%s" % "\n",
                      "#SBATCH -p skx%s" % "\n",
                      "#SBATCH -t 01:20:00%s" % "\n",
                      'echo "SCRIPT $PE_HOSTFILE"%s' % "\n",
                      "module load gcc/13.2.0%s" % "\n",
                      "module load impi/21.11%s" % "\n",
                      "module load meep/1.28%s" % "\n",
                      "source ~/.bashrc%s" % "\n",
                      "conda activate ndo%s" % "\n",
                      #"echo new_file: %s %s" % (new_file, "\n"),
                      #"echo air_raw_path: %s %s" % (air_raw_path, "\n"),
                      #"echo air_data_path: %s %s" % (air_data_path, "\n"),
                      #"echo metal_raw_path: %s %s" % (metal_raw_path, "\n"),
                      #"echo metal_data_path: %s %s" % (metal_data_path, "\n"),
                      #"echo ticker_file: %s %s" % (ticker_file, "\n"),
                      #"ibrun -np 4 meep no-metal?=true theta_deg=%s %s | tee %s%s" % (theta_deg, new_file, air_raw_path, "\n"),
                      "ibrun -np 32 meep no-metal?=true sy=%s theta_deg=%s %s | tee %s;%s" % (cell_size, theta_deg, new_file, air_raw_path, "\n"),
                      #"meep no-metal?=true theta_deg=%s %s | tee %s;%s" % (theta_deg, new_file, air_raw_path, "\n"),
                      "grep flux1: %s > %s;%s" % (air_raw_path, air_data_path, "\n"),
                      #"ibrun -np 4 meep sr=%s ht=%s sy=%s theta_deg=%s %s |tee %s;%s" % (sr, ht, cell_size, theta_deg, new_file, metal_raw_path, "\n"),
                      "mpirun -np 32 meep no-metal?=false sr=%s ht=%s sy=%s theta_deg=%s %s |tee %s;%s" % (sr, ht, cell_size, theta_deg, new_file, metal_raw_path, "\n"),
                      #"meep sr=%s ht=%s sy=%s theta_deg=%s %s |tee %s;%s" % (sr, ht, cell_size, theta_deg, new_file, metal_raw_path, "\n"),
                      "grep flux1: %s > %s;%s" % (metal_raw_path, metal_data_path, "\n"),
                      "echo %s;%s" % (info_file, "\n"),
                      #"wait;%s" % ("\n"),
                      "python %s %s;%s" % (main_home_dir + "NanoDotOptimization/optimize-ag-dot-angle-evaluate.py", info_file, "\n"),
                      "rm -r %s %s" % (ticker_file, "\n"),
                      "echo 1 >> %s %s" % (ticker_file, "\n")

                      ])
    
    file1.close()

    sleep(15)  # Pause to give time for simulation file to be created
    os.system("ssh login1 sbatch " + sbatch_file)  # Execute the simulation file

    success = 0

    #(4) Extracting Data From optimization
    max_time = (100*100)
    #max_time = (50)
    time_count = 0
    # Wait for data to be stable and ready for processing
    while success == 0:
        try:
            a = open(ticker_file, "r").read()
            a = int(a)
            if a == 1:
                printing(f"files pass:{(air_data_path, metal_data_path)}")
                success = 1   
        except:
            pass
        
        if time_count == max_time:
                raise Exception(f"ticker not existing: {ticker_file}")

        time_count = time_count + 1
        time.sleep(1)
        
    os.system("ssh login1 rm -r " +
            ticker_file + " " +
            air_raw_path + " " +
            metal_raw_path + " " +
            metal_data_path + " " +
            air_data_path + " " +
            new_file + " " +
            main_home_dir + "ag-dot-angle" + code + "* " +
            file_home_path + "ag-dot-angle" + code + "* ")

    printing(f"finished deleting files; code: {code}")

    # (9) Returning of Result and Continuity of Optimization
    printing(f'Executed: {filename}')
    
    return filename

def get_values(x: [float], param: str):

    sr = x[0]
    ht = x[1]
    cs = x[2]
    #cs = 0.001 * 250
    theta_deg = x[3]
    #theta_deg = x[2]


    filename = make_filename(sr, ht, cs, theta_deg)

    log_answer = check_log(filename, param)
    if len(log_answer) > 0:
        printing(f'Referenced: {filename}')
        return log_answer[0]
    else:
        obj_func_run(x)
        return check_log(filename, param)[0]

def b(x: [float]):

    return get_values(x, "b-param")

def c(x: [float]):

    return get_values(x, "c-param")

def b_var(x: [float]):

    #return get_values(x, "b_var")
    return get_values(x, "b_var")


def c_var(x: [float]):

    return get_values(x, "c_var")


def c_upper_constraint(x: [float]): return 5 - get_values(x, "c-param")
def c_lower_constraint(x: [float]): return get_values(x, "c-param")

def b_lower_constraint(x: [float]): return get_values(x, "b-param") - 1  # b-param should be >= 1
def b_upper_constraint(x: [float]): return 50 - get_values(x, "b-param")  # b-param should be <= 60

'''
def c_constraint(x: [float]):

    return 15 - get_values(x, "c-param")
'''

def b_var_constraint(x: [float]):

    return 10 - get_values(x, "b_var")


#bounds = {'sr': (0.001 * 5, 0.001 * 125), 'ht': (0.001 * 50, 0.001 * 100), 'cs': (0.001 * 25, 0.001 * 250), 'theta_deg': (0.0, 0.0)}# Bounds for optimization

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
    .add_function(c_var)
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

    max_evaluations = 160
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

    df1 = pd.read_csv(main_work_dir + "ag-dot-angle-pretraining-unpruned.csv")

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
        #print(f'(Solution #{sol + 1}): Variables={front[sol].variables}; Objectives={front[sol].objectives}')
        printing(f'(Solution #{sol + 1}): (Filename - {make_filename(float(vars[0]), float(vars[1]), float(vars[2]), float(vars[3]))})')
        printing(f'             Variables={vars}')
        printing(f'             Objectives={front[sol].objectives}')

    printing(f"Computing time: {algorithm.total_computing_time}")


