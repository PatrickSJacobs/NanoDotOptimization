
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
from multiprocessing import Pool
#from multiprocessing import cpu_count
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
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations

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
logging_file = file_home_path + "calc_log_obj.csv"

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
    df = pd.read_csv(logging_file)
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
    #printing((sr, ht, cs, theta_deg))

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
                      "#SBATCH -n 48%s" % "\n",
                      "#SBATCH -N 1%s" % "\n",
                      "#SBATCH --mail-user=pjacobs7@eagles.nccu.edu%s" % "\n",
                      "#SBATCH --mail-type=all%s" % "\n",
                      "#SBATCH -p skx%s" % "\n",
                      "#SBATCH -t 05:20:00%s" % "\n",
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
                      "ibrun -np 48 meep no-metal?=true sy=%s theta_deg=%s %s | tee %s;%s" % (cell_size, theta_deg, new_file, air_raw_path, "\n"),
                      #"meep no-metal?=true theta_deg=%s %s | tee %s;%s" % (theta_deg, new_file, air_raw_path, "\n"),
                      "grep flux1: %s > %s;%s" % (air_raw_path, air_data_path, "\n"),
                      #"ibrun -np 4 meep sr=%s ht=%s sy=%s theta_deg=%s %s |tee %s;%s" % (sr, ht, cell_size, theta_deg, new_file, metal_raw_path, "\n"),
                      "mpirun -np 48 meep no-metal?=false sr=%s ht=%s sy=%s theta_deg=%s %s |tee %s;%s" % (sr, ht, cell_size, theta_deg, new_file, metal_raw_path, "\n"),
                      #"meep sr=%s ht=%s sy=%s theta_deg=%s %s |tee %s;%s" % (sr, ht, cell_size, theta_deg, new_file, metal_raw_path, "\n"),
                      "grep flux1: %s > %s;%s" % (metal_raw_path, metal_data_path, "\n"),
                      "echo %s;%s" % (info_file, "\n"),
                      #"wait;%s" % ("\n"),
                      "python %s %s;%s" % (main_home_dir + "NanoDotOptimization/optimize-ag-dot-angle-evaluate.py", info_file, "\n"),
                      "rm -r %s %s" % (ticker_file, "\n"),
                      "echo 1 >> %s %s" % (ticker_file, "\n")

                      ])
    
    file1.close()

    sleep(10)  # Pause to give time for simulation file to be created
    printing(x)
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
                #printing(f"files pass:{(air_data_path, metal_data_path)}")
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

    #printing(f"finished deleting files; code: {code}")

    # (9) Returning of Result and Continuity of Optimization
    #printing(f'Executed: {filename}')
    
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
        #printing(f'Referenced: {filename}')
        return log_answer[0]
    else:
        obj_func_run(x)
        return check_log(filename, param)[0]
    
def runtest(x: [float]):

    b = get_values(x, "b-param")
    c = get_values(x, "c-param")
    b_var = get_values(x, "b_var")
    
    printing(f'(Solution: (Filename - {make_filename(float(x[0]), float(x[1]), float(x[2]), float(x[3]))})')
    printing(f'             Variables={x}')
    printing(f'             Objectives={[b, c, b_var]}')
    printing('\n')

if __name__ == "__main__":

    with open(logging_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "sr", "ht", "cs", "theta_deg", "b-param", "c-param", "b_var", "c_var","execution time", "step count"])
        file.close()
    
    test_population = [
        np.array([0.0157, 0.0602, 0.4537, 0.0]),
        np.array([0.0177, 0.0602, 0.4537, 0.0]),
        np.array([0.0167, 0.0612, 0.4537, 0.0]),
        np.array([0.0167, 0.0592, 0.4537, 0.0]),
        np.array([0.0167, 0.0602, 0.4547, 0.0]),
        np.array([0.0167, 0.0602, 0.4527, 0.0]),
        np.array([0.0167, 0.0602, 0.4537, 0.0]),
        np.array([0.0157, 0.0592, 0.4547, 0.0])
        ]

    # Determine the number of available CPU cores for parallel processing
    #NUM_PROCESSES = cpu_count()

    #print(f"NUM_PROCESSES: {NUM_PROCESSES}")
    #with Pool(processes=NUM_PROCESSES) as pool:
    with Pool(processes=len(test_population)) as pool:
        # Compute integrals for all data points in parallel
        int1 = np.array(pool.map(runtest, test_population))
    
    os.system("scancel -u tg881088")
