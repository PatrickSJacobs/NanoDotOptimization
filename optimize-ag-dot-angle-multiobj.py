
from jmetal.core.problem import OnTheFlyFloatProblem
from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.comparator import DominanceComparator
from datetime import datetime
import time
import string
import random
import csv
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import statistics
from scipy.signal import find_peaks

# File and directory paths
current_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
main_home_dir = "/home1/08809/tg881088/"
folder_name = f"opt_{str(current_time)}"
file_home_path = os.path.join(main_home_dir, f"{folder_name}_processed/")
main_work_dir = "/work2/08809/tg881088/"
file_work_path = os.path.join(main_work_dir, f"{folder_name}_raw/")
progress_file = os.path.join(file_home_path, "progress.txt")

# Make directories for the optimization files
os.makedirs(file_home_path, exist_ok=True)
os.makedirs(file_work_path, exist_ok=True)

# Initialize progress file
with open(progress_file, 'w') as file_naught:
    file_naught.write(f"Beginning optimization\n")

def printing(msg: str):
    with open(progress_file, 'a') as file_printer:
        file_printer.write(f"{msg}\n")
    print(msg)

def check_log(filename: str, param: str):
    df = pd.read_csv(os.path.join(file_home_path, "calc_log_obj.csv"))
    return df.loc[df['filename'] == filename, param].tolist()

def make_filename(type: str, sr: float, ht: float, cs: float, theta_deg: float) -> str:
    display_theta_deg = str(round(theta_deg if theta_deg > 0 else theta_deg + 360.0, 1)).replace(".", "_")
    return f"{folder_name}_{type}_sr_{str(round(sr * 10000, 1)).replace('.', '_')}nm_ht_{str(round(ht * 10000, 1)).replace('.', '_')}nm_cs_{str(round(cs * 10000, 1)).replace('.', '_')}nm_theta_deg_{display_theta_deg}"

def obj_func_calc(wvls, R_meep):
    K = 2
    xs = wvls[:-K]
    ys = R_meep[:-K]

    mam = max(ys)
    maxis = [y for i, y in enumerate(ys) if y >= 0.5 * mam]
    ind = [i for i, y in enumerate(ys) if y >= 0.5 * mam]
    ind = list(range(min(ind), max(ind) + 1))
    maxis = [ys[i] for i in ind]

    neg_ys = [(-y + mam) for y in ys]
    peaks, _ = find_peaks([(-y + mam) for y in maxis])

    maxi = xs[ys.index(mam)]
    if peaks:
        real_min = min([ys[ind[peak]] for peak in peaks], default=mam)
        index_list = [i for i, y in enumerate(ys) if y >= real_min]
        ys_fixed = [ys[i] for i in index_list]
        L = [wvl for wvl, freq in zip([xs[i] for i in index_list], np.ceil(np.array(ys_fixed) / np.min(ys_fixed))) for _ in range(int(freq))]
        maxi = statistics.mean(L)

    def objective(x, b, c):
        maximum = np.array([maxi] * len(x))
        return 1 / (c ** 2 * (1 + (10000 / b * (x - maximum)) ** 2))

    popt, popv = curve_fit(objective, xs, ys)
    b, c = popt
    b_var, c_var = popv[0][0], popv[1][1]

    printing("finished obj_eval")
    return b, c ** 2 * 10 - 10, b_var * 100, c_var * 100

def sim(run_file, filenames=[], input_lines=[]):
    executable = open(main_home_dir + "NanoDotOptimization/ag-dot-angle0.txt", 'r')
    lines = input_lines + executable.readlines()
    code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    new_name = os.path.join(file_home_path, f"ag-dot-angle{code}.py")

    with open(new_name, 'w') as file0:
        file0.writelines(lines)
    
    ticker_file = os.path.join(file_home_path, f"ticker{code}.txt")
    with open(ticker_file, 'w') as file2:
        file2.write("0")

    sbatch_file = os.path.join(file_home_path, f"{run_file}.txt")
    air_sim_file = os.path.join(file_home_path, f"ag-dot-angle_{filenames[0]}")
    metal_sim_file = os.path.join(file_home_path, f"ag-dot-angle_{filenames[1]}")

    sbatch_content = [
        "#!/bin/bash\n",
        "#SBATCH -J myMPI\n",
        "#SBATCH -o myMPI.%j.o\n",
        "#SBATCH -n 32\n",
        "#SBATCH -N 1\n",
        "#SBATCH --mail-user=pjacobs7@eagles.nccu.edu\n",
        "#SBATCH --mail-type=all\n",
        "#SBATCH -p skx\n",
        "#SBATCH -t 00:30:00\n",
        'echo "SCRIPT $PE_HOSTFILE"\n',
        "source ~/.bashrc\n",
        "conda activate ndo\n",
        f"mpirun -np 32 python -m mpi4py {new_name} True | tee -a {air_sim_file}.out ; grep flux1: {air_sim_file}.out > {air_sim_file}.dat;\n",
        f"mpirun -np 32 python -m mpi4py {new_name} False | tee -a {metal_sim_file}.out ; grep flux1: {metal_sim_file}.out > {metal_sim_file}.dat;\n",
        "conda deactivate\n",
        f"rm -r {ticker_file}\n",
        f"echo 1 >> {ticker_file}\n",
    ]

    with open(sbatch_file, 'w') as file1:
        file1.writelines(sbatch_content)

    time.sleep(15)
    os.system(f"ssh login1 sbatch {sbatch_file}")

    return ticker_file, f"{air_sim_file}.out", f"{air_sim_file}.dat", f"{metal_sim_file}.out", f"{metal_sim_file}.dat", new_name

def obj_func_run(x: [float]):
    sr, ht, theta_deg = x[0], x[1], x[2]
    #cs = 0.001 * 250
    cs = 0.001 * 125
    cell_size = 2 * sr + cs
    printing((sr, ht, cs, theta_deg))

    input_lines = [
        "#----------------------------------------\n",
        f"sr = {sr}\n",
        f"ht = {ht}\n",
        f"sy = {cell_size}\n",
        f"theta_deg = {theta_deg}\n",
        "#----------------------------------------\n"
    ]

    filename = make_filename("", sr, ht, cs, theta_deg)
    ticker_file, air_raw_path, air_data_path, metal_raw_path, metal_data_path, new_name = sim(
        run_file=filename, 
        filenames=[make_filename(name, sr, ht, cs, theta_deg) for name in ["air", "metal"]],
        input_lines=input_lines
    )

    max_time = 70 * 60
    time_count, success2 = 0, 0

    while success2 == 0 and time_count < max_time:
        try:
            with open(ticker_file, "r") as tick_file:
                tick = int(tick_file.read())
                if tick == 1:
                    printing(f"files pass:{(air_data_path, metal_data_path)}")
                    success2 = 1
        except:
            time.sleep(1)
            time_count += 1

    if os.path.isfile(metal_data_path) and os.path.isfile(air_data_path):
        b, c, b_var, c_var = 0.001, 15, 10, 10
        try:
            df = pd.read_csv(metal_data_path, header=None)
            df0 = pd.read_csv(air_data_path, header=None)

            printing("success2 df")

            wvls = df[1] * 0.32
            R_meep = [abs(-r / r0) for r, r0 in zip(df[3], df0[3])]
            wvls, R_meep = wvls[:-2], R_meep[:-2]

            with open(os.path.join(file_work_path, f"calc_log_data_step_{len(open(os.path.join(file_home_path, 'calc_log_obj.csv'), 'r').readlines())}_{filename}.csv"), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["wvl", "refl"])
                writer.writerows(zip(wvls, R_meep))

            printing("passed to obj")
            b, c, b_var, c_var = obj_func_calc(wvls, R_meep)
            printing("came out of obj")

        except Exception as e:
            printing(f"Error: {e}")
            os.system(f"ssh login1 rm -r {ticker_file} {air_raw_path} {air_data_path} {new_name}* {metal_raw_path} {metal_data_path}")
        finally:
            with open(os.path.join(file_home_path, "calc_log_obj.csv"), 'a') as file:
                writer = csv.writer(file)
                writer.writerow([filename, sr, ht, cs, theta_deg, b, c, b_var, c_var, datetime.now().strftime("%m_%d_%Y__%H_%M_%S"), len(open(os.path.join(file_home_path, "calc_log_obj.csv"), 'r').readlines())])

        time.sleep(10)
        os.system(f"ssh login1 rm -r {ticker_file} {air_raw_path} {air_data_path} {new_name}* {metal_raw_path} {metal_data_path}")

        printing(f"finished deleting files; code: {metal_raw_path}")
        printing(f'Executed: {filename}')
        return filename

def get_values(x: [float], param: str):
    filename = make_filename("", x[0], x[1], x[2], 0.001 * 250)
    log_answer = check_log(filename, param)
    if log_answer:
        printing(f'Referenced: {filename}')
        return log_answer[0]
    else:
        time.sleep(5)
        return check_log(obj_func_run(x), param)[0]

def b(x: [float]): return get_values(x, "b-param")
def c(x: [float]): return get_values(x, "c-param")
def b_var(x: [float]): return get_values(x, "b_var")
def c_var(x: [float]): return get_values(x, "c_var")

# Double-sided constraints for b, c, b_var, and c_var parameters
def b_lower_constraint(x: [float]): return get_values(x, "b-param") - 1  # b-param should be >= 1
def b_upper_constraint(x: [float]): return 150 - get_values(x, "b-param")  # b-param should be <= 15

def c_lower_constraint(x: [float]): return get_values(x, "c-param") - 1  # c-param should be >= 1
def c_upper_constraint(x: [float]): return 1.5 - get_values(x, "c-param")  # c-param should be <= 1.5

def b_var_lower_constraint(x: [float]): return get_values(x, "b_var") - 0  # b_var should be >= 0
def b_var_upper_constraint(x: [float]): return 10 - get_values(x, "b_var")  # b_var should be <= 10

# Double-sided constraints for the decision variables
#def sr_lower_constraint(x: [float]): return x[0] - 0.001 * 5  # sr should be >= 0.001 * 5
#def sr_upper_constraint(x: [float]): return 0.001 * 125 - x[0]  # sr should be <= 0.001 * 125

#def ht_lower_constraint(x: [float]): return x[1] - 0.001 * 50  # ht should be >= 0.001 * 50
#def ht_upper_constraint(x: [float]): return 0.001 * 100 - x[1]  # ht should be <= 0.001 * 100

# Define the optimization problem with double-sided constraints
problem = (
    OnTheFlyFloatProblem()
    .set_name("Testing")
    .add_variable(0.001 * 5, 0.001 * 125)
    .add_variable(0.001 * 50, 0.001 * 100)
    .add_variable(0.0, 0.0)
    .add_function(c)
    .add_function(b)
    .add_function(b_var)
    .add_function(c_var)
    .add_constraint(b_lower_constraint)
    .add_constraint(b_upper_constraint)
    .add_constraint(c_lower_constraint)
    .add_constraint(c_upper_constraint)
    .add_constraint(b_var_lower_constraint)
    .add_constraint(b_var_upper_constraint)
    #.add_constraint(sr_lower_constraint)
    #.add_constraint(sr_upper_constraint)
    #.add_constraint(ht_lower_constraint)
    #.add_constraint(ht_upper_constraint)
)

# Main execution
if __name__ == "__main__":
    with open(os.path.join(file_home_path, "calc_log_obj.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "sr", "ht", "cs", "theta_deg", "b-param", "c-param", "b_var", "c_var", "execution time", "step_count"])

    #max_evaluations = 640
    max_evaluations = 50

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
    
    for sol in range(len(front)):
        vars = front[sol].variables
        filename = make_filename("", float(vars[0]), float(vars[1]), float(vars[2]), float(vars[3]))
        printing(f'(Solution #{sol + 1}): (Filename - {filename})')
        printing(f'             Variables={vars}')
        printing(f'             Objectives={front[sol].objectives}')

    printing(f"Computing time: {algorithm.total_computing_time}")
