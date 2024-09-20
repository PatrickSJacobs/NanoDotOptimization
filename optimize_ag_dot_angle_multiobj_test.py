import os
import csv
from datetime import datetime
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

from ag_dot_angle_obj import obj_func_run

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
from botorch.models import MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import sample_simplex
from botorch.models.transforms import Standardize, Normalize
from botorch import fit_gpytorch_model
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.models import MultiOutputGP, FixedNoiseMultiOutputGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood

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


# Your existing code for file paths and utility functions remains the same
# ...

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

    # Bounds (include 'cs' bounds)
    bounds = torch.tensor([
        [0.005, 0.05, 0.25, 0.0],   # Lower bounds for sr, ht, cs, theta_deg
        [0.125, 0.1, 0.25, 90.0]     # Upper bounds for sr, ht, cs, theta_deg
    ], dtype=torch.double)

    # Normalize the training inputs
    train_X_normalized = (train_X - bounds[0]) / (bounds[1] - bounds[0])

    num_iterations = 4  # Number of optimization iterations

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

    n_tasks = train_Y.shape[-1]
    printing(f"n_tasks: {n_tasks}")

    for iteration in range(num_iterations):
        # Normalize inputs
        train_X_normalized = (train_X - bounds[0]) / (bounds[1] - bounds[0])

        # Fit the multi-output GP model
        # We will use MultiOutputGP with an IndexKernel to model correlations between outputs

        # Create a Multitask Gaussian Likelihood
        likelihood = MultitaskGaussianLikelihood(num_tasks=n_tasks)

        # Define the model
        model = MultiOutputGP(
            train_X=train_X_normalized,
            train_Y=train_Y,
            likelihood=likelihood,
            outcome_transform=Standardize(m=n_tasks),
        )

        # Define the marginal log likelihood
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit the model
        fit_gpytorch_model(mll)

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
        ref_point = feasible_Y.min(dim=0).values - 0.1 * (feasible_Y.max(dim=0).values - feasible_Y.min(dim=0).values)
        ref_point = ref_point.tolist()
        printing(f"ref_point: {ref_point}")

        # Define the acquisition function using qNEHVI
        sampler = SobolQMCNormalSampler(num_samples=128)
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X_normalized,
            constraints=constraints,
            sampler=sampler,
            prune_baseline=True,
        )

        # Optimize the acquisition function to get the next candidate
        candidate_normalized, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([torch.zeros(train_X.shape[-1]), torch.ones(train_X.shape[-1])]),
            q=1,
            num_restarts=5,
            raw_samples=20,  # For initialization
        )

        # Unnormalize the candidate
        candidate = candidate_normalized * (bounds[1] - bounds[0]) + bounds[0]

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