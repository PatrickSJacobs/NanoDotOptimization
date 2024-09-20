import os
import csv
import random
import string
import time
from datetime import datetime
from time import sleep
import numpy as np
import pandas as pd
import torch

# Import necessary modules from BoTorch and GPyTorch
from botorch.models.transforms import Standardize
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP

current_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")  # Getting the current time
main_home_dir = "/home1/08809/tg881088/"  # Home directory for optimization
folder_name = "opt_%s" % str(current_time)  # Folder name for optimization files
file_home_path = main_home_dir + folder_name + "_processed/"  # Folder name for optimization files
main_work_dir = "/work2/08809/tg881088/"  # Home directory for optimization
file_work_path = main_work_dir + folder_name + "_raw/"  # Folder name for optimization files
progress_file = file_home_path + "progress.txt"
os.mkdir(file_home_path)  # Making folder name for optimization files
os.mkdir(file_work_path)  # Making folder name for data log
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
    theta_deg = x[3]
    filename = make_filename(sr, ht, cs, theta_deg)
    log_answer = check_log(filename, param)
    if len(log_answer) > 0:
        printing(f'Referenced: {filename}')
        return log_answer[0]
    else:
        obj_func_run(x)
        return check_log(filename, param)[0]

# Define constraints as functions (accepting posterior samples Y)
def c1(samples):
    return 5 - samples[..., 0]  # c-param <= 5

def c2(samples):
    return samples[..., 1] - 1  # b-param >= 1

def c3(samples):
    return 50 - samples[..., 1]  # b-param <= 50

def c4(samples):
    return 10 - samples[..., 2]  # b_var <= 10

constraints = [c1, c2, c3, c4]

# Define the function to evaluate the candidate
def evaluate_candidate(candidate):
    x = candidate.squeeze(0).tolist()
    x_input = [x[0], x[1], x[2], x[3]]
    y0 = get_values(x_input, 'c-param')
    y1 = get_values(x_input, 'b-param')
    y2 = get_values(x_input, 'b_var')
    y3 = get_values(x_input, 'c_var')
    y = torch.tensor([[y0, y1, y2, y3]], dtype=torch.double)
    return y

if __name__ == "__main__":
    # Load pretraining data from CSV file
     # Initialize CSV log file if it doesn't exist
    calc_log_obj_path = os.path.join(file_home_path, "calc_log_obj.csv")
    if not os.path.exists(calc_log_obj_path):
        with open(calc_log_obj_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["filename", "sr", "ht", "cs", "theta_deg", "b-param", "c-param", "b_var", "c_var", "execution time",
                 "step count"])
            
            
    pretraining_data_path = main_work_dir + 'ag-dot-angle-pretraining.csv'  # Replace with your CSV file path
    df = pd.read_csv(pretraining_data_path)

    # Inputs (include 'cs' since it's now a variable)
    train_X = torch.tensor(df[['sr', 'ht', 'cs', 'theta_deg']].values, dtype=torch.double)
    print(len(train_X))

    # Outputs
    train_Y = torch.tensor(df[['c-param', 'b-param', 'b_var', 'c_var']].values, dtype=torch.double)
    print(len(train_Y))

    task_indices = torch.arange(4, dtype=torch.long).repeat(train_X.shape[0])

    # Repeat each train_X row for each task
    train_X = train_X.repeat(4, 1)

    # Reshape train_Y to have shape n x 1, where n includes all tasks
    train_Y = train_Y.T.flatten().view(-1, 4)  # Reshape train_Y to n x 1

    
    #train_Y = train_Y.reshape(-1, 1)

    # Bounds (include 'cs' bounds)
    bounds = torch.tensor([
        [0.005, 0.05, 0.25, 0.0],   # Lower bounds for sr, ht, cs, theta_deg
        [0.125, 0.1, 0.25, 90.0]     # Upper bounds for sr, ht, cs, theta_deg
    ], dtype=torch.double).T

    num_iterations = 4  # Number of optimization iterations

    num_tasks = train_Y.shape[1]  # Should be 4

    printing(f"num_tasks: {num_tasks}") 
    for iteration in range(num_iterations):
        # Fit the GP model
        
        model = SaasFullyBayesianMultiTaskGP(
            train_X=torch.cat([train_X, task_indices.unsqueeze(-1)], dim=-1),
            train_Y=train_Y,
            task_feature=-1  # The task feature is now the last column
        )
        
        #fit_fully_bayesian_model_nuts(model)
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=128,  # Reduce from 512 to 128
            num_samples=128,   # Reduce from 256 to 128
            thinning=16,       # Adjust thinning if necessary
            disable_progbar=False  # You can set this to True to disable the progress bar
        )

        #posterior = mtsaas_gp.posterior(test_X)

        # Get standardized training outputs
        train_Y_std = model.outcome_transform(train_Y)[0]

        # Compute feasibility mask
        is_feasible = (c1(train_Y_std) >= 0) & (c2(train_Y_std) >= 0) & (c3(train_Y_std) >= 0) & (c4(train_Y_std) >= 0)
        is_feasible = is_feasible.all(dim=-1)

        if is_feasible.sum() == 0:
            printing("No feasible observations found.")
            break

        feasible_Y = train_Y_std[is_feasible]

        # Define reference point for hypervolume calculation
        printing("No reference point")
        ref_point = feasible_Y.min(dim=0).values - 0.1 * (feasible_Y.max(dim=0).values - feasible_Y.min(dim=0).values)
        ref_point = ref_point.tolist()
        printing(f"ref_point: {ref_point}")
        # Remove partitioning (not needed for qNEHVI)
        # partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point), Y=feasible_Y)

        # Define the acquisition function using qNEHVI
        sampler = SobolQMCNormalSampler(num_samples=128)
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X,
            constraints=constraints,
            sampler=sampler,
            prune_baseline=True,
        )

        # Optimize the acquisition function to get the next candidate
        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,  # For initialization
        )

        # Evaluate the candidate
        y_new = evaluate_candidate(candidate)

        # Update training data
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y_new], dim=0)

        # Print progress
        printing(f"Iteration {iteration + 1}/{num_iterations}")
        printing(f"Candidate: {candidate}")
        printing(f"Objective values: {y_new}")

    # After optimization, process the results
    # Compute feasibility mask for final train_Y
    is_feasible = (c1(train_Y) >= 0) & (c2(train_Y) >= 0) & (c3(train_Y) >= 0) & (c4(train_Y) >= 0)
    is_feasible = is_feasible.all(dim=-1)

    # Get feasible train_Y and train_X
    feasible_Y = train_Y[is_feasible]
    feasible_X = train_X[is_feasible]

    # Extract the Pareto front
    pareto_mask = is_non_dominated(feasible_Y)
    pareto_front = feasible_Y[pareto_mask]
    pareto_points = feasible_X[pareto_mask]

    # Save Pareto front to CSV
    pareto_df = pd.DataFrame(
        torch.cat([pareto_points, pareto_front], dim=1).numpy(),
        columns=['sr', 'ht', 'cs', 'theta_deg', 'c-param', 'b-param', 'b_var', 'c_var']
    )
    pareto_df.to_csv(os.path.join(file_home_path, 'pareto_front.csv'), index=False)