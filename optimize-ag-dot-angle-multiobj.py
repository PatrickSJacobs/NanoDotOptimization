
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

def make_filename(type, sr, ht, cs, theta_deg):
    display_theta_deg = str(round(theta_deg if theta_deg > 0 else theta_deg + 360.0,
                                  1)).replace(".", "_")  # angle to be used

    filename = "%s_%s_sr_%s_ht_%s_cs_%s_theta_deg_%s" % (str(folder_name),
                                                      type,
                                                      str(round(sr * 10000, 1)).replace(".", "_") + "nm",
                                                      str(round(ht * 10000, 1)).replace(".", "_") + "nm",
                                                      str(round(cs * 10000, 1)).replace(".", "_") + "nm",
                                                      display_theta_deg,
                                                      )  # filename to be used
    return filename

def obj_func_calc(wvls, R_meep):
    '''
    (6) Objective Function Execution
        - Fit distribution of reflectance to a multimodal distribution
        - Get means, variances and heights of n subdistributions
        - Calculate vector peak_eval = [norm(mean(1), variance(1), 1/height(1)), norm(mean(2), variance(2), 1/height(2)), ..., norm(mean(n), variance(n), 1/height(n))]
        - Return sum of all elements in peak_eval
    '''

    # Make histogram based on reflectance data
    K = 2

    xs = wvls
    ys = R_meep

    xs = xs[: len(xs) - K]
    ys = ys[: len(ys) - K]

    mam = max(ys)
    ind = []
    maxis = []
    ty = len(ys)
    for i in range(ty):
        j = ys[i]
        if j >= 0.5 * mam:
            maxis += [j]
            ind += [i]

    ind = [i for i in range(min(ind), max(ind) + 1)]
    maxis = [ys[i] for i in ind]

    neg_ys_prime = [element + mam for element in [(-1) * y for y in maxis]]
    neg_ys = [element + mam for element in [(-1) * y for y in ys]]

    peaks, _ = find_peaks(neg_ys_prime)
    maxi = 0

    if len(peaks) > 0:
        neg_min = mam
        real_min = 0

        for peak in peaks:
            if neg_min >= neg_ys[ind[peak]]:
                neg_min = neg_ys[ind[peak]]
                real_min = ys[ind[peak]]

        index_list = [i for i in range(len(ys)) if ys[i] >= real_min]

        ys_fixed = [ys[i] for i in index_list]

        L = []
        for (wvl, freq) in zip([xs[i] for i in index_list], np.ceil(np.array(ys_fixed) / np.min(ys_fixed))):
            L += [wvl for i in range(int(freq))]

        maxi = statistics.mean(L)

    else:
        maxi = xs[ys.index(mam)]

    # tsr = sum(x * i for i, x in enumerate(L, 1)) / len(L)

    # print(tsr)

    q = 1000

    # 1/(1 + (q/b*(x - a))^2)/c == 1/(pc)

    def objective(x, b, c):
        # maxi = np.array([xs[ys.index(mam)] for i in range(len(x))])
        # maxi = sum(u * i for i, u in enumerate(x, 1)) / len(x)
        maximum = np.array([maxi for i in x])

        #f = q / (1 + (q / b * (x - maximum)) ** 2) / c
        f = 1 / (c**2*(1 + (q / b * (x - maximum)) ** 2))

        return f

    popt, popv = curve_fit(objective, xs, ys)

    b, c = popt
    b_var = popv[0][0]
    c_var = popv[1][1]

    printing("finished obj_eval")

    return b, c**2 * 10 - 10, b_var * 100, c_var * 100

def sim(filename="make_filename(sr, ht, cs, theta_deg)", input_lines=[]):

    '''
    inputlines = [";----------------------------------------%s" % "\n",
                  "(define-param sr %s)%s" % (sr, "\n"),
                  "(define-param ht %s)%s" % (ht, "\n"),
                  "(define-param sy %s)%s" % (cell_size, "\n"),
                  "(define-param theta_deg %s)%s" % (theta_deg, "\n"),
                  ";----------------------------------------%s" % "\n"
                  ]
    '''

    executable = open(main_home_dir + "NanoDotOptimization/ag-dot-angle0.ctl", 'r')
    lines = input_lines + executable.readlines()
    code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    new_file = file_home_path + "ag-dot-angle" + code + ".ctl"
    file0 = open(new_file, 'w')
    file0.writelines(lines)
    file0.close()

    ticker_file = file_home_path + "ticker" + code + ".txt"
    file2 = open(ticker_file, 'w')
    file2.write("0")
    file2.close()

    sbatch_name = file_home_path + "/" + str(filename)
    sbatch_file = sbatch_name + ".txt"
    
    file1 = open(sbatch_file, 'w')

    sim_file = "%sag-dot-angle_%s" % (file_home_path, filename)
    raw_path = sim_file + ".out"
    data_path = sim_file + ".dat"

    file1.writelines(["#!/bin/bash%s" % "\n",
                      "#SBATCH -J myMPI%s" % "\n",
                      "#SBATCH -o myMPI.%s%s" % ("o%j", "\n"),
                      #"#SBATCH -n 32%s" % "\n",
                    "#SBATCH -n 1%s" % "\n",
                      "#SBATCH -N 1%s" % "\n",
                      "#SBATCH --mail-user=pjacobs7@eagles.nccu.edu%s" % "\n",
                      "#SBATCH --mail-type=all%s" % "\n",
                      "#SBATCH -p skx%s" % "\n",
                     # "#SBATCH -t 01:10:00%s" % "\n",
                        "#SBATCH -t 00:10:00%s" % "\n",

                      'echo "SCRIPT $PE_HOSTFILE"%s' % "\n",
                      "module load gcc/13.2.0%s" % "\n",
                      "module load impi/21.11%s" % "\n",
                      "module load meep/1.28%s" % "\n",
                     # "mpirun -np 32 meep %s |tee %s;%s" % (new_file, raw_path, "\n"),
                      #"mpirun -np 1 meep %s |tee %s;%s" % (new_file, raw_path, "\n"),
                      "meep %s |tee %s;%s" % (new_file, raw_path, "\n"),
                      "grep flux1: %s > %s%s" % (raw_path, data_path, "\n"),
                      "rm -r %s %s" % (ticker_file, "\n"),
                      "echo 1 >> %s %s" % (ticker_file, "\n")

                      ])

    file1.close()

    sleep(15)  # Pause to give time for simulation file to be created 
    os.system("ssh login1 sbatch " + sbatch_file + " > " + sbatch_name + "_job_id.txt")  # Execute the simulation file
    #os.system("ssh login1 sbatch /home1/08809/tg881088/NanoDotOptimization/testing.txt")  # Execute the simulation file

    return (ticker_file, raw_path, data_path, main_home_dir + "ag-dot-angle" + code, file_home_path + "ag-dot-angle" + code, sbatch_name + "_job_id.txt")


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
    #cs = x[2]
    #theta_deg = x[3]
    cs = 0.001 * 250
    theta_deg = x[2]
    cell_size = 2 * sr + cs

    #sleep(10)# Sleep to give code time to process parallelization

    # Parameters to be used in current evaluation
    printing((sr, ht, cs, theta_deg))
    filename = make_filename("", sr, ht, cs, theta_deg)

    filename0 = make_filename("air", sr, ht, cs, theta_deg)

    input_lines0 = [";----------------------------------------%s" % "\n",
                  "(define-param sr %s)%s" % (sr, "\n"),
                  "(define-param ht %s)%s" % (ht, "\n"),
                  "(define-param sy %s)%s" % (cell_size, "\n"),
                  "(define-param theta_deg %s)%s" % (theta_deg, "\n"),
                  "(define-param no-metal false)",
                  ";----------------------------------------%s" % "\n"
                  ]



    ticker_file0, air_raw_path, air_data_path, main_del0, home_del0, jobfile0 = sim(filename=filename0, input_lines=input_lines0)
    #= None, None, None, None, None, None

    print("air " + ticker_file0)
    sleep(100) 

    filename1 = make_filename("metal", sr, ht, cs, theta_deg)


    input_lines1 = [";----------------------------------------%s" % "\n",
                    "(define-param sr %s)%s" % (sr, "\n"),
                    "(define-param ht %s)%s" % (ht, "\n"),
                    "(define-param sy %s)%s" % (cell_size, "\n"),
                    "(define-param theta_deg %s)%s" % (theta_deg, "\n"),
                    "(define-param no-metal true)",
                    ";----------------------------------------%s" % "\n"
                    ]

    ticker_file1, metal_raw_path, metal_data_path, main_del1, home_del1, jobfile1 = None, None, None, None, None, None

    #ticker_file1, metal_raw_path, metal_data_path, main_del1, home_del1, jobfile1 = sim(filename=filename1, input_lines=input_lines1)

    print("metal " + ticker_file1)


    success = 0

    #(4) Extracting Data From optimization
    max_time = (70*60)
    time_count = 0
    # Wait for data to be stable and ready for processing
    while success == 0:
        try:
            tick1 = open(ticker_file0, "r").read()
            tick2 = open(ticker_file1, "r").read()
            if os.path.isfile(metal_data_path) and os.path.isfile(air_data_path):
                tick1 = int(tick1)
                tick2 = int(tick2)
                if tick1 == 1 & tick2 == 1:
                    printing(f"files pass:{(air_data_path, metal_data_path)}")
                    success = 1
        except:
            if time_count == max_time:
                raise Exception(f"ticker not existing: {ticker_file0}")
            else:
                pass

        time_count = time_count + 1
        time.sleep(1)

    # Check if data is good and data file exists, if not error
    if os.path.isfile(metal_data_path) and os.path.isfile(air_data_path):
        df = None
        df0 = None
        b, c, b_var, c_var = None, None, None, None
        log_obj_exe = open(file_home_path + "calc_log_obj.csv", 'r').readlines()
        step_count = int(len(log_obj_exe))
        
        try:    
            df = pd.read_csv(metal_data_path, header=None)
            df0 = pd.read_csv(air_data_path, header=None)

            printing("success df")

            # Get wavelengths and reflectance data
            wvls = df[1] * 0.32
            R_meep = [np.abs(- r / r0) for r, r0 in zip(df[3], df0[3])]

            wvls = wvls[: len(wvls) - 2]
            R_meep = R_meep[: len(R_meep) - 2]

          

            with open(file_work_path + "calc_log_data_step_%s_%s.csv" % (step_count, filename), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["wvl", "refl"])
                for (wvl, refl) in zip(wvls, R_meep):
                    writer.writerow([wvl, refl])
            printing("passed to obj")
            # (5) Sending Data Through Objective function

            b, c, b_var, c_var = obj_func_calc(wvls, R_meep)

            printing("came out of obj")

        except Exception as e:
            print(f"Error : {e}")
            print((air_data_path, metal_data_path, ticker_file0))
            sleep(1)
            
            
            # Define the path to the file that contains the job ID
            # Open the file and read the content
            with open(jobfile0, "r") as file:
                # Read the first line
                line = file.readline().strip()

                # Split the line and extract the job ID
                # The line format is "Submitted batch job <job_id>"
                job_id = line.split()[-1]  # Extract the last word, which is the job ID
                os.system(f"scancel {job_id}")
                print(f"Successfully cancelled job {job_id}")


            with open(jobfile1, "r") as file:
                # Read the first line
                line = file.readline().strip()

                # Split the line and extract the job ID
                # The line format is "Submitted batch job <job_id>"
                job_id = line.split()[-1]  # Extract the last word, which is the job ID
                os.system(f"scancel {job_id}")

            os.system("ssh login1 rm -r " +
                            ticker_file0 + " " +
                            air_raw_path + " " +
                            air_data_path + " " +
                            main_del0 + "* " +
                            home_del0 + "* " +
                            ticker_file1 + " " +
                            metal_raw_path + " " +
                            metal_data_path + " " +
                            main_del1 + "* " +
                            home_del1 + "* " +
                            jobfile0 + " " +
                            jobfile1 + " "
                            )

            b, c, b_var, c_var = 0.001, 15, 10, 10

            


        # (7) Logging of Current Data
        with open(file_home_path + "calc_log_obj.csv", 'a') as file:
            writer = csv.writer(file)
            writer.writerow([filename, sr, ht, cs, theta_deg, b, c, b_var, c_var, datetime.now().strftime("%m_%d_%Y__%H_%M_%S"), step_count])
            file.close()

        #execution_dictionary[filename] = {"b": b, "c": c, "b_var": b_var, "c_var": c_var}

    else:
        printing(f"({metal_data_path}): It isn't a file!")
        raise ValueError(f"({metal_data_path}): It isn't a file!")

    # (8) Deleting Excess Files
    sleep(10)
    os.system("ssh login1 rm -r " +
              ticker_file0 + " " +
              air_raw_path + " " +
              air_data_path + " " +
              main_del0 + "* " +
              home_del0 + "* " +
              ticker_file1 + " " +
              metal_raw_path + " " +
              metal_data_path + " " +
              main_del1 + "* " +
              home_del1 + "* "
              )

    printing(f"finished deleting files; code: {metal_raw_path}")

    # (9) Returning of Result and Continuity of Optimization
    printing(f'Executed: {filename}')
    return filename


def get_values(x: [float], param: str):

    sr = x[0]
    ht = x[1]
    #cs = x[2]
    cs = 0.001 * 250
    #theta_deg = x[3]
    theta_deg = x[2]


    filename = make_filename("", sr, ht, cs, theta_deg)

    log_answer = check_log(filename, param)
    if len(log_answer) > 0:
        printing(f'Referenced: {filename}')
        return log_answer[0]
    else:
        obj_func_run(x)
        sleep(5)
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


def b_constraint(x: [float]):

    return 1 - get_values(x, "b-param")


def c_constraint(x: [float]):

    return 15 - get_values(x, "c-param")


def b_var_constraint(x: [float]):

    return 10 - get_values(x, "b_var")


#bounds = {'sr': (0.001 * 5, 0.001 * 125), 'ht': (0.001 * 50, 0.001 * 100), 'cs': (0.001 * 25, 0.001 * 250), 'theta_deg': (0.0, 0.0)}# Bounds for optimization

problem = (
    OnTheFlyFloatProblem()
    .set_name("Testing")
    .add_variable(0.001 * 5, 0.001 * 125)
    .add_variable(0.001 * 50, 0.001 * 100)
    #.add_variable(0.001 * 25, 0.001 * 250)
    #.add_variable(0.001 * 25, 1.0)
    .add_variable(0.0, 0.0)
    .add_function(c)
    .add_function(b)
    .add_function(b_var)
    .add_function(c_var)
    .add_constraint(b_constraint)
    .add_constraint(c_constraint)
    .add_constraint(b_var_constraint)
)

if __name__ == "__main__":

    with open(file_home_path + "calc_log_obj.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "sr", "ht", "cs", "theta_deg", "b-param", "c-param", "b_var", "c_var","execution time", "step count"])
        file.close()

    max_evaluations = 640

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

    algorithm = GDE3(
        population_evaluator=MultiprocessEvaluator(processes=16),
        problem=problem,
        population_size=1,
        cr=0.9,
        f=0.8,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        dominance_comparator=DominanceComparator(),
    )


    algorithm.run()
    solutions = algorithm.get_result()
    front = algorithm.get_result()
    #front = get_non_dominated_solutions(solutions)

    for sol in range(len(front)):
        vars = front[sol].variables
        #print(f'(Solution #{sol + 1}): Variables={front[sol].variables}; Objectives={front[sol].objectives}')
        printing(f'(Solution #{sol + 1}): (Filename - {make_filename("", float(vars[0]), float(vars[1]), float(vars[2]), float(vars[3]))})')
        printing(f'             Variables={vars}')
        printing(f'             Objectives={front[sol].objectives}')

    printing(f"Computing time: {algorithm.total_computing_time}")


