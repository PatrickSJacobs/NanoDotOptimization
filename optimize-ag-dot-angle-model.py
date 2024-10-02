from datetime import datetime
import time
import string
import random
from time import sleep
import csv
import numpy as np
import os
import warnings
import pandas as pd

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

# execution_dictionary = {}

def printing(string):
    with open(progress_file, 'r') as f:
        file_printout = f.readlines()
    lines = file_printout + [f"{str(string)}\n"]
    with open(progress_file, 'w') as f:
        f.writelines(lines)
    print(string)
    pass

def check_log(filename: str):
    df = pd.read_csv(file_home_path + "calc_log_obj.csv")
    # Assuming 'calc_log_obj.csv' has columns that can be converted to a dictionary
    # Here, we can return the first matching row as a dict
    row = df.loc[df['filename'] == filename]
    if not row.empty:
        return row.to_dict(orient='records')[0]
    else:
        return {}

def make_filename(sr, ht, cs, theta_deg):
    display_theta_deg = str(round(theta_deg if theta_deg > 0 else theta_deg + 360.0,
                                  1)).replace(".", "_")  # angle to be used

    filename = "%s_sr_%s_ht_%s_cs_%s_theta_deg_%s" % (
        str(folder_name),
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
    #theta_deg = x[3]
    theta_deg = 0.0
    # cs = 0.001 * 250
    # theta_deg = x[2]

    #sleep(10)  # Sleep to give code time to process parallelization

    # Parameters to be used in current evaluation
    # printing((sr, ht, cs, theta_deg))

    filename = make_filename(sr, ht, cs, theta_deg)

    # Creating Scheme executable for current optimization; ag-dot-angle0.ctl cannot be used simultaneously with multiple workers)
    executable_path = os.path.join(main_home_dir, "NanoDotOptimization", "ag-dot-angle0.ctl")
    with open(executable_path, 'r') as f:
        lines = f.readlines()
    code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    new_name = file_home_path + "ag-dot-angle" + code
    new_file = new_name + ".ctl"
    with open(new_file, 'w') as f:
        f.writelines(lines)

    # Creating ticker file to make sure the data is created and stable for processing
    ticker_file = file_home_path + "ticker" + code + ".txt"
    with open(ticker_file, 'w') as f:
        f.write("0")

    # Creation of simulation "subjob" file
    sbatch_file = os.path.join(file_home_path, f"{filename}.txt")
    with open(sbatch_file, 'w') as f:
        cell_size = 2 * sr + cs
        air_file = f"{file_home_path}air-angle_{filename}"
        metal_file = f"{file_home_path}ag-dot-angle_{filename}"
        
        air_raw_path = f"{air_file}.out"
        metal_raw_path = f"{metal_file}.out"
        air_data_path = f"{air_file}.dat"
        metal_data_path = f"{metal_file}.dat"
        
        info_file = f"{new_name}.txt"

        # Write info file
        with open(info_file, "w") as info_f:
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
                info_f.write(f"{item}\n")

        # Write sbatch script
        f.writelines([
            "#!/bin/bash\n",
            "#SBATCH -J myMPI\n",
            "#SBATCH -o myMPI.o%j\n",
            "#SBATCH -n 48\n",
            "#SBATCH -N 1\n",
            "#SBATCH --mail-user=pjacobs7@eagles.nccu.edu\n",
            "#SBATCH --mail-type=all\n",
            "#SBATCH -p skx\n",
            "#SBATCH -t 04:20:00\n",
            'echo "SCRIPT $PE_HOSTFILE"\n',
            "module load gcc/13.2.0\n",
            "module load impi/21.11\n",
            "module load meep/1.28\n",
            "source ~/.bashrc\n",
            "conda activate ndo\n",
            # Execute simulation commands
            f"ibrun -np 48 meep no-metal?=true sy={cell_size} theta_deg={theta_deg} {new_file} | tee {air_raw_path};\n",
            f"grep flux1: {air_raw_path} > {air_data_path};\n",
            f"ibrun -np 48 meep no-metal?=false sr={sr} ht={ht} sy={cell_size} theta_deg={theta_deg} {new_file} | tee {metal_raw_path};\n",
            f"grep flux1: {metal_raw_path} > {metal_data_path};\n",
            f"echo {info_file};\n",
            f"python {os.path.join(main_home_dir, 'NanoDotOptimization', 'optimize-ag-dot-angle-evaluate.py')} {info_file};\n",
            f"rm -r {ticker_file} {air_raw_path} {metal_raw_path} {metal_data_path} {air_data_path} {new_file} {main_home_dir}ag-dot-angle{code}* {file_home_path}ag-dot-angle{code}* \n",
            f"echo 1 >> {ticker_file} \n"
        ])

    #sleep(10)  # Pause to give time for simulation file to be created
    printing(x)
    os.system("ssh login1 sbatch " + sbatch_file)  # Execute the simulation file

    success = 0

    # (4) Extracting Data From optimization
    max_time = 60*60  # or 10000
    time_count = 0
    # Wait for data to be stable and ready for processing
    while success == 0:
        try:
            with open(ticker_file, "r") as f:
                a = int(f.read())
            if a == 1:
                success = 1
        except:
            pass

        if time_count == max_time:
            raise Exception(f"ticker not existing: {ticker_file}")

        time_count += 1
        time.sleep(10)

    # Remove temporary files
    os.system("ssh login1 rm -r " +
              ticker_file + " " +
              air_raw_path + " " +
              metal_raw_path + " " +
              metal_data_path + " " +
              air_data_path + " " +
              new_file + " " +
              main_home_dir + "ag-dot-angle" + code + "* " +
              file_home_path + "ag-dot-angle" + code + "* ")

    # (9) Returning of Result and Continuity of Optimization
    return filename

import multiprocessing as mp

def get_values_single(xi_list):
    sr = xi_list[0]
    ht = xi_list[1]
    cs = xi_list[2]
    theta_deg = 0.0
    #theta_deg = xi_list[3]

    filename = make_filename(sr, ht, cs, theta_deg)
    log_answer = check_log(filename)
    if len(log_answer) > 0:
        obj_run = log_answer
    else:
        obj_func_run(xi_list)
        obj_run = check_log(filename)

    return obj_run["b-param"], obj_run["c-param"], obj_run["b_var"]

import torch

def get_values(x: torch.Tensor):
    xi_list_batch = x.tolist()  # Convert tensor to list of lists
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(get_values_single, xi_list_batch)

    # Unpack results
    b_param_list, c_param_list, b_var_list = zip(*results)

    # Convert lists to tensors
    b_param_tensor = torch.tensor(b_param_list, **tkwargs)
    c_param_tensor = torch.tensor(c_param_list, **tkwargs)
    b_var_tensor = torch.tensor(b_var_list, **tkwargs)

    # Return as a dictionary
    return {
        "b-param": b_param_tensor,
        "c-param": c_param_tensor,
        "b_var": b_var_tensor
    }

with open(file_home_path + "calc_log_obj.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "sr", "ht", "cs", "theta_deg", "b-param", "c-param", "b_var", "c_var", "execution time", "step count"])
    file.close()

################################################################################################################################################################


import botorch.settings
from botorch.exceptions import BadInitialCandidatesWarning  # Import outside functions
from botorch import fit_gpytorch_mll  # Import outside functions

# ### Set dtype and device
botorch.settings.debug(True)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda:3" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# ### Load Custom Initial Data

def load_initial_data(file_path):
    """
    Loads initial training data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: (train_x, train_obj)
            - train_x (torch.Tensor): Tensor of shape (n_samples, d).
            - train_obj (torch.Tensor): Tensor of shape (n_samples, M).
    """
    df = pd.read_csv(file_path)
    #required_columns = ["sr", "ht", "cs", "theta_deg", 'b-param', 'c-param', 'b_var']
    required_columns = ["sr", "ht", "cs", 'b-param', 'c-param', 'b_var']

    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")

    train_x = torch.tensor(df[["sr", "ht", "cs"]].values, **tkwargs)
    train_obj = torch.tensor(df[['b-param', 'c-param', 'b_var']].values, **tkwargs)

    # **Negate the objectives to convert from minimization to maximization**
    train_obj = -train_obj  # Assuming CSV has objectives for minimization

    return train_x, train_obj


# Define the parameter bounds (adjust as per your problem)
'''
bounds = torch.tensor([
    [0.01, 0.01, 0.001 * 25],
    #[0.001 * 125, 0.001 * 125, 0.001 * 400, 0.0001]
    [0.001 * 125, 0.001 * 125, 0.001 * 125]
], **tkwargs)
'''

# Path to your initial data CSV file
initial_data_path = main_work_dir + "ag-dot-angle-pretraining.csv"  # Replace with your actual file path
# Load the initial data
train_x_initial, train_obj_initial = load_initial_data(initial_data_path)

# Calculate bounds for train_x_initial before normalization
train_x_min = list(torch.min(train_x_initial, dim=0).values)
train_x_max = list(torch.max(train_x_initial, dim=0).values)


bounds = torch.tensor([
    train_x_min,
    train_x_max
], **tkwargs)
'''
bounds = torch.tensor([
    [0.001, 0.001, 0.001],
    [1.0, 1.0, 1.0],
], **tkwargs)
'''

from botorch.utils.transforms import normalize, unnormalize

train_x_initial = normalize(train_x_initial, bounds)

# Sanity check
if torch.any(train_x_initial < 0) or torch.any(train_x_initial > 1):
    raise ValueError("Normalized input features are out of bounds [0, 1].")
else:
    print("All input features are within the [0, 1] range.")

print(f"Initial training data shape: {train_x_initial.shape}, {train_obj_initial.shape}")

# ### Define the Optimization Problem

class NanoDotProblem:
    def __init__(self, norm_bounds, num_objectives, ref_point, penalty=1e6, **tkwargs):
        #self.bounds = bounds
        self.bounds = norm_bounds
        self.num_objectives = num_objectives
        self.dim = norm_bounds.shape[1]
        # Define your reference point (must be worse than any feasible objective value)
        self.ref_point = torch.tensor(ref_point, **tkwargs)
        # Set maximum hypervolume if known (optional)
        self.max_hv = None  # Replace with actual value if known
        self.penalty = penalty  # Penalty factor

    def __call__(self, x):
        """
        Evaluates the objectives at x with penalties for violations.

        Args:
            x (torch.Tensor): Tensor of shape (..., self.dim)

        Returns:
            torch.Tensor: Tensor of shape (..., self.num_objectives)
        """
        # Retrieve the objective values
        
        print(x)
        obj_run = get_values(x)

        # Define the objectives (negated for maximization)
        obj1 = -obj_run["b-param"]
        obj2 = -obj_run["c-param"]
        obj3 = -obj_run["b_var"]

        return torch.stack([obj1, obj2, obj3], dim=-1)

# ### Compute Reference Point

def compute_reference_point(train_obj, margin=0.05):
    """
    Computes the reference point based on initial objective values.

    Args:
        train_obj (torch.Tensor): Tensor of shape (n_samples, M).
        margin (float): Fractional margin to set the reference point slightly worse.

    Returns:
        list: Reference point values for each objective.
    """
    # Find the minimum observed objective values (since objectives are negated for minimization)
    min_obj = train_obj.min(dim=0).values  # Shape: (M,)

    # Apply margin to set the reference point slightly worse
    # Since we are maximizing obj, "worse" means lower values
    ref_point = (min_obj - margin * torch.abs(min_obj)).tolist()

    return ref_point


# Define the optimization problem
problem = NanoDotProblem(norm_bounds=bounds, num_objectives=3, ref_point=compute_reference_point(train_obj_initial, margin=0.05))

# ### Initialize the Hypervolume Calculator

from botorch.utils.multi_objective.hypervolume import Hypervolume  # Import outside functions

hv = Hypervolume(ref_point=problem.ref_point)
print(f"problem.ref_point: {problem.ref_point}")
print(f"hv: {hv}")

# ### Compute Initial Hypervolume

from botorch.utils.multi_objective.pareto import is_non_dominated  # Import outside functions

def compute_hypervolume(train_obj, hv):
    """
    Computes the hypervolume of the Pareto front.

    Args:
        train_obj (torch.Tensor): Tensor of shape (n_samples, M).
        hv (Hypervolume): Hypervolume calculator instance.

    Returns:
        float: Computed hypervolume.
    """
    if train_obj.shape[0] == 0:
        return 0.0

    # Identify Pareto optimal points
    pareto_mask = is_non_dominated(train_obj)
    pareto_y = train_obj[pareto_mask]

    if pareto_y.shape[0] == 0:
        return 0.0

    # Compute hypervolume
    volume = hv.compute(pareto_y)
    return volume

# Compute initial hypervolume for qNEHVI
initial_hv_qnehvi = compute_hypervolume(train_obj_initial, hv)
hvs_qnehvi = [initial_hv_qnehvi]
print(f"Initial hypervolume for qNEHVI: {initial_hv_qnehvi}")

# ### Initialize GP Models

from botorch.models.gp_regression import SingleTaskGP  # Import outside functions
from botorch.models.model_list_gp_regression import ModelListGP  # Import outside functions
from botorch.models.transforms.outcome import Standardize  # Import outside functions
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood  # Import outside functions
from botorch.utils.transforms import normalize  # Import outside functions

def initialize_model(train_x, train_obj, problem):
    """
    Initializes a multi-output GP model for the objectives.

    Args:
        train_x (torch.Tensor): Tensor of shape (n_samples, d).
        train_obj (torch.Tensor): Tensor of shape (n_samples, M).
        problem (NanoDotProblem): The optimization problem instance.

    Returns:
        tuple: (mll, model)
            - mll (gpytorch.mlls.SumMarginalLogLikelihood): Marginal log likelihood.
            - model (ModelListGP): Multi-output GP model.
    """
    # Sanity check to ensure all inputs are within [0, 1]
    if torch.any(train_x < 0) or torch.any(train_x > 1):
        raise ValueError("Normalized input features are out of bounds [0, 1].")
    else:
        print("All input features are within the [0, 1] range.")

    # Define GP models for each objective
    models = []
    for i in range(train_obj.shape[-1]):
        models.append(
            SingleTaskGP(
                train_x, train_obj[..., i:i+1], outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# Initialize models for qNEHVI
mll_qnehvi, model_qnehvi = initialize_model(train_x_initial, train_obj_initial, problem)

# ### Define the Acquisition Function Helper

from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement  # Import outside functions
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective  # Import outside functions
from botorch.sampling.normal import SobolQMCNormalSampler  # Import outside functions
from botorch.optim.optimize import optimize_acqf  # Import outside functions
from botorch.utils.transforms import unnormalize  # Import outside functions

def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler, problem, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES):
    """
    Optimizes the qNEHVI acquisition function and returns new candidate points and their evaluations.

    Args:
        model (ModelListGP): The surrogate GP model.
        train_x (torch.Tensor): Tensor of shape (n_samples, d).
        train_obj (torch.Tensor): Tensor of shape (n_samples, M).
        sampler (SobolQMCNormalSampler): Sampler for Monte Carlo integration.
        problem (NanoDotProblem): The optimization problem instance.
        BATCH_SIZE (int): Number of candidates to generate.
        NUM_RESTARTS (int): Number of restarts for acquisition optimization.
        RAW_SAMPLES (int): Number of raw samples for acquisition optimization.

    Returns:
        tuple: (new_x, new_obj)
            - new_x (torch.Tensor): Tensor of shape (BATCH_SIZE, d).
            - new_obj (torch.Tensor): Tensor of shape (BATCH_SIZE, M).
    """
    # Define the qNEHVI acquisition function
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),
        X_baseline=train_x,
        sampler=sampler,
        prune_baseline=True,
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(problem.num_objectives))),
        # constraints=constraints_list,  # Constraints removed
    )

    # Define bounds in normalized space
    standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
    standard_bounds[1] = 1
    
    print("made acq func")

    # Optimize the acquisition function to find new candidates
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for initialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    print("optimized acq func")
    # Unnormalize the candidates to original space
    new_x = unnormalize(candidates.detach(), problem.bounds)
    print(f"new sol: {new_x}")
    # Evaluate objectives at new candidates
    new_obj = problem(new_x)
    print(f"New objectives: {new_obj}")

    # Check for inf or nan in new_obj
    if not torch.isfinite(new_obj).all():
        print("Warning: new_obj contains inf or nan values.")
        # Remove data points with inf or nan values
        valid_indices = torch.isfinite(new_obj).all(dim=1)
        new_x = new_x[valid_indices]
        new_obj = new_obj[valid_indices]
        if new_obj.shape[0] == 0:
            raise ValueError("All new objective values contain inf or nan.")

    return new_x, new_obj

# ### Perform Bayesian Optimization Loop with qNEHVI Only

# Suppress specific warnings
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define BO parameters
#N_BATCH = 20 if not SMOKE_TEST else 1
N_BATCH = 20 if not SMOKE_TEST else 1
MC_SAMPLES = 128 if not SMOKE_TEST else 16
#BATCH_SIZE = 2
BATCH_SIZE = 16
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
verbose = True

# Define the sampler
qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

# Initialize batch numbers for coloring
batch_number_qnehvi = np.arange(train_obj_initial.shape[0])  # Initial batch numbers

for iteration in range(1, N_BATCH + 1):
    t0 = time.monotonic()
    print(f"Iteration {iteration}")

    # Fit the qNEHVI model using fit_gpytorch_mll
    try:
        fit_gpytorch_mll(mll_qnehvi)
    except Exception as e:
        print(f"Model fitting failed at iteration {iteration}: {e}")
        break

   
    '''
    # Optimize acquisition function and get new observations
    try:
        new_x_qnehvi, new_obj_qnehvi = optimize_qnehvi_and_get_observation(
            model_qnehvi, train_x_initial, train_obj_initial, qnehvi_sampler, problem, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES
        )
    except Exception as e:
        print(f"Acquisition optimization failed at iteration {iteration}: {e}")
        break 
    '''
    # Optimize acquisition function and get new observations
    new_x_qnehvi, new_obj_qnehvi = optimize_qnehvi_and_get_observation(
            model_qnehvi, train_x_initial, train_obj_initial, qnehvi_sampler, problem, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES
        )

    # Update training data for qNEHVI
    # Normalize the new candidates
    normalized_new_x = normalize(new_x_qnehvi, problem.bounds)
    train_x_initial = torch.cat([train_x_initial, normalized_new_x])
    train_obj_initial = torch.cat([train_obj_initial, new_obj_qnehvi])

    # Update batch numbers for qNEHVI
    new_batch_numbers_qnehvi = np.full((BATCH_SIZE,), iteration)
    batch_number_qnehvi = np.concatenate([batch_number_qnehvi, new_batch_numbers_qnehvi])

    # Compute hypervolume for qNEHVI
    volume_qnehvi = compute_hypervolume(train_obj_initial, hv)
    hvs_qnehvi.append(volume_qnehvi)

    # Reinitialize the qNEHVI model for the next iteration
    mll_qnehvi, model_qnehvi = initialize_model(
        train_x_initial, train_obj_initial, problem
    )

    t1 = time.monotonic()

    if verbose:
        print(
            f"\nBatch {iteration:>2}: Hypervolume (qNEHVI) = "
            f"{volume_qnehvi:>4.2f}, "
            f"time = {t1-t0:>4.2f} seconds.",
            end="",
        )
    else:
        print(".", end="")

# ### Extract and Save the Final Pareto Front

import pandas as pd  # Ensure pandas is imported at the top

def extract_pareto_front(train_x, train_obj):
    """
    Extracts the Pareto front from the training data.

    Args:
        train_x (torch.Tensor): Tensor of shape (n_samples, d) representing input parameters.
        train_obj (torch.Tensor): Tensor of shape (n_samples, M) representing objective values.

    Returns:
        tuple: (pareto_x, pareto_front)
            - pareto_x: Tensor containing input parameters of Pareto optimal points.
            - pareto_front: Tensor containing Pareto optimal objective values.
    """
    if train_obj.shape[0] == 0:
        print("No feasible solutions found.")
        empty_tensor = torch.empty(0, train_obj.shape[-1])
        return empty_tensor, empty_tensor

    # Identify Pareto optimal points
    pareto_mask = is_non_dominated(train_obj)
    pareto_front = train_obj[pareto_mask]
    pareto_x = train_x[pareto_mask]

    return pareto_x, pareto_front

def save_pareto_front(pareto_x, pareto_front, filename="pareto_front_qnehvi.csv"):
    """
    Saves the Pareto front to a CSV file, including both parameters and objectives.

    Args:
        pareto_x (torch.Tensor): Tensor containing normalized input parameters of Pareto optimal points.
        pareto_front (torch.Tensor): Tensor containing Pareto optimal objective values.
        norm_bounds (torch.Tensor): Tensor containing normalization bounds.
        filename (str): Name of the file to save the Pareto front.
    """
    if pareto_front.numel() == 0:
        print("No Pareto front to save.")
        return

    # Re-negate objectives to original minimization scale
    pareto_front_original = -pareto_front  # Since obj = -f

    # Unnormalize parameters
    pareto_x_original = unnormalize(pareto_x, problem.bounds)

    # Convert tensors to numpy arrays
    pareto_x_np = pareto_x_original.cpu().numpy()
    pareto_front_np = pareto_front_original.cpu().numpy()

    # Combine parameters and objectives into a single DataFrame
    data = np.hstack((pareto_x_np, pareto_front_np))
    
    columns = (
        [f"Parameter_{i+1}" for i in range(pareto_x_np.shape[1])]
        + [f"Objective_{i+1}" for i in range(pareto_front_np.shape[1])]
    )
    
    #columns = (["sr", "ht", "cs", "theta_deg", 'b-param', 'c-param', 'b_var'])
    
    df_pareto = pd.DataFrame(data, columns=columns)
    df_pareto.to_csv(filename, index=False)
    print(f"Pareto front saved to {filename}.")

# Extract Pareto front for qNEHVI
pareto_x_qnehvi, pareto_front_qnehvi = extract_pareto_front(train_x_initial, train_obj_initial)

if pareto_front_qnehvi.numel() > 0:
    print("\nFinal Pareto Front for qNEHVI:")
    # Unnormalize parameters
    pareto_x_qnehvi_original = unnormalize(pareto_x_qnehvi, problem.bounds).cpu().numpy()
    pareto_front_qnehvi_np = -pareto_front_qnehvi.cpu().numpy()  # Re-negate to original minimization
    for idx, (x_vals, obj_vals) in enumerate(zip(pareto_x_qnehvi_original, pareto_front_qnehvi_np), start=1):
        print(f"Pareto Point {idx}: Parameters: {x_vals}, Objectives: {obj_vals}")
    # Save to a file
    save_pareto_front(pareto_x_qnehvi, pareto_front_qnehvi)
else:
    print("\nNo Pareto front found for qNEHVI.")

# ### Plot the Observations Colored by Iteration and Overlay Pareto Front

import matplotlib.pyplot as plt  # Import outside functions
from matplotlib.cm import ScalarMappable  # Import outside functions
import matplotlib  # Ensure matplotlib is imported

# Ensure the non-interactive backend is set
matplotlib.use('Agg')

fig, ax = plt.subplots(figsize=(8, 6))  # Single plot for qNEHVI

cm = plt.get_cmap("viridis")

# Plot qNEHVI Observations
# Since it's multi-objective, it's better to plot a pair or use dimensionality reduction
# For simplicity, we can plot the first two objectives

scatter = ax.scatter(
    -train_obj_initial[:, 0].cpu().numpy(),
    -train_obj_initial[:, 1].cpu().numpy(),
    c=batch_number_qnehvi,
    alpha=0.8,
    cmap=cm,
    label="qNEHVI Observations"
)
ax.set_title("qNEHVI Observations")
ax.set_xlabel("Objective 1")
ax.set_xlim(-15, 15)  # Adjust based on your objective ranges
ax.set_ylim(-15, 15)
ax.set_ylabel("Objective 2")

# Overlay Pareto front if exists
if pareto_front_qnehvi.numel() > 0:
    pareto_front_np = -pareto_front_qnehvi.cpu().numpy()  # Re-negate to original minimization
    ax.scatter(
        pareto_front_np[:, 0],
        pareto_front_np[:, 1],
        marker='*',
        color='red',
        s=200,
        label='Pareto Front'
    )

# Create a colorbar
norm = plt.Normalize(batch_number_qnehvi.min(), batch_number_qnehvi.max())
sm = ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.ax.set_title("Iteration")

# Add legends
ax.legend()

# Adjust layout
plt.tight_layout()

# Save the figure to the local directory
plt.savefig("observations_colored_by_iteration_with_pareto.png", bbox_inches='tight')
plt.close()
