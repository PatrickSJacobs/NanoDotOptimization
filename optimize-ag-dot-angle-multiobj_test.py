from ag_dot_angle_obj import obj_func_run, current_time, main_home_dir, folder_name, file_home_path, main_work_dir, file_work_path, progress_file
import os
import csv
import pandas as pd
import sys

def printing(string):
    with open(progress_file, 'a') as file_printer:
        file_printer.write(f"{str(string)}\n")
    print(string)

def check_log(filename: str, param: str):
    df = pd.read_csv(os.path.join(file_home_path, "calc_log_obj.csv"))
    return df.loc[df['filename'] == filename, param].tolist()

def make_filename(sr, ht, cs, theta_deg):
    display_theta_deg = str(round(theta_deg if theta_deg > 0 else theta_deg + 360.0, 1)).replace(".", "_")
    filename = "%s_sr_%s_ht_%s_cs_%s_theta_deg_%s" % (
        str(folder_name),
        str(round(sr * 10000, 1)).replace(".", "_") + "nm",
        str(round(ht * 10000, 1)).replace(".", "_") + "nm",
        str(round(cs * 10000, 1)).replace(".", "_") + "nm",
        display_theta_deg,
    )
    return filename

import torch

from botorch.utils.multi_objective.pareto import is_non_dominated

from botorch.models import MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize as InputNormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

def get_values(x: [float], param: str):
    sr, ht, cs, theta_deg = x
    filename = make_filename(sr, ht, cs, theta_deg)
    log_answer = check_log(filename, param)
    if log_answer:
        printing(f'Referenced: {filename}')
        return log_answer[0]
    else:
        obj_func_run(x)
        return check_log(filename, param)[0]

# Define constraints as functions (accepting posterior samples Y)
def c1(samples):
    return samples[..., 0]  # c-param <= 5

def c2(samples):
    return samples[..., 1] # b-param >= 1

def c3(samples):
    return samples[..., 1]  # b-param <= 50

def c4(samples):
    return samples[..., 2]  # b_var <= 10

constraints = [c1, c2, c3, c4]

# Define the function to evaluate the candidate
def evaluate_candidate(candidate):
    x = candidate.squeeze(0).tolist()
    x_input = [x[0], x[1], x[2], x[3]]
    y0 = get_values(x_input, 'c-param')
    y1 = get_values(x_input, 'b-param')
    y2 = get_values(x_input, 'b_var')
    y = torch.tensor([[y0, y1, y2]], dtype=torch.double)
    return y

if __name__ == "__main__":
    # Load pretraining data from CSV file
    calc_log_obj_path = os.path.join(file_home_path, "calc_log_obj.csv")
    if not os.path.exists(calc_log_obj_path):
        with open(calc_log_obj_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["filename", "sr", "ht", "cs", "theta_deg", "b-param", "c-param", "b_var", "c_var", "execution time",
                 "step count"])

    pretraining_data_path = os.path.join(main_work_dir, 'ag-dot-angle-pretraining.csv')  # Replace with your CSV file path
    df = pd.read_csv(pretraining_data_path)

    # Inputs
    train_X = torch.tensor(df[['sr', 'ht', 'cs', 'theta_deg']].values, dtype=torch.double)

    # Outputs
    train_Y = torch.tensor(df[['c-param', 'b-param', 'b_var']].values, dtype=torch.double)

    # Bounds
    bounds = torch.tensor([
        [0.005, 0.05, 0.025, 0.0],   # Lower bounds for sr, ht, cs, theta_deg
        [0.125, 0.1, 0.25, 90.0]     # Upper bounds for sr, ht, cs, theta_deg
    ], dtype=torch.double)

    # Normalize the training inputs using InputNormalize
    input_transform = InputNormalize(d=4, bounds=bounds)
    train_X_normalized = input_transform(train_X)

    num_iterations = 4  # Number of optimization iterations

    # Define the number of tasks (objectives)
    num_tasks = train_Y.shape[-1]
    printing(f"Number of tasks: {num_tasks}")

    # Prepare the training data for MultiTaskGP
    task_indices = torch.arange(num_tasks, dtype=torch.long)  # [0, 1, 2]
    tasks = task_indices.unsqueeze(0).repeat(train_X_normalized.shape[0], 1)  # [N, num_tasks]

    # Expand train_X_normalized to include task indices
    train_X_expanded = train_X_normalized.unsqueeze(1).repeat(1, num_tasks, 1)  # [N, num_tasks, D]
    train_X_expanded = torch.cat([train_X_expanded, tasks.unsqueeze(-1)], dim=-1)  # [N, num_tasks, D+1]
    train_X_expanded = train_X_expanded.view(-1, train_X_expanded.shape[-1])  # [N*num_tasks, D+1]

    # Expand train_Y accordingly
    train_Y_expanded = train_Y.transpose(0, 1).reshape(-1, 1)  # [N*num_tasks, 1]

    task_feature = train_X_expanded.shape[1] - 1  # Index of the task feature

    for iteration in range(num_iterations):
        # Initialize and fit the MultiTaskGP model
        model = MultiTaskGP(
            train_X=train_X_expanded,
            train_Y=train_Y_expanded,
            task_feature=task_feature,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        print("Model fitted")

        # Compute feasibility mask using model predictions
        with torch.no_grad():
            posterior = model.posterior(train_X_expanded)
            mean = posterior.mean.view(-1, num_tasks)

        is_feasible = (c1(mean) >= 0) & (c2(mean) >= 0) & (c3(mean) >= 0) & (c4(mean) >= 0)
        is_feasible = is_feasible.all(dim=-1)

        if is_feasible.sum() == 0:
            printing("No feasible observations found.")
            break

        feasible_Y = mean[is_feasible]
        feasible_X = train_X_normalized[is_feasible]

        # Extract Pareto-optimal points to use as baseline
        pareto_mask = is_non_dominated(feasible_Y)
        pareto_Y = feasible_Y[pareto_mask]
        pareto_X = feasible_X[pareto_mask]

        # Limit the number of baseline points to avoid high dimensionality
        max_baseline_points = 50
        if pareto_X.shape[0] > max_baseline_points:
            indices = torch.randperm(pareto_X.shape[0])[:max_baseline_points]
            X_baseline = pareto_X[indices]
        else:
            X_baseline = pareto_X

        # Expand X_baseline to include task indices
        tasks_baseline = task_indices.unsqueeze(0).repeat(X_baseline.shape[0], 1)  # [B, num_tasks]
        X_baseline_expanded = X_baseline.unsqueeze(1).repeat(1, num_tasks, 1)  # [B, num_tasks, D]
        X_baseline_expanded = torch.cat([X_baseline_expanded, tasks_baseline.unsqueeze(-1)], dim=-1)  # [B, num_tasks, D+1]
        X_baseline_expanded = X_baseline_expanded.view(-1, X_baseline_expanded.shape[-1])  # [B*num_tasks, D+1]

        # Define reference point for hypervolume calculation
        ref_point = feasible_Y.min(dim=0).values - 0.1 * (feasible_Y.max(dim=0).values - feasible_Y.min(dim=0).values)
        ref_point = ref_point.tolist()
        printing(f"ref_point: {ref_point}")

        # Define the acquisition function using qNEHVI
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline_expanded,
            #constraints=constraints,
            sampler=sampler,
            prune_baseline=True,
            cache_root=False,
        )

        # Optimize the acquisition function to get the next candidate
        candidate_expanded, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([
                torch.zeros(train_X_expanded.shape[-1]),
                torch.ones(train_X_expanded.shape[-1]),
            ]),
            q=1,
            num_restarts=5,
            raw_samples=20,  # For initialization
        )

        # Extract candidate without task indices
        candidate_normalized = candidate_expanded[..., :-1]

        # Denormalize the candidate
        candidate = input_transform.untransform(candidate_normalized)

        # Evaluate the candidate
        y_new = evaluate_candidate(candidate)

        # Update training data
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y_new], dim=0)

        # Normalize the new candidate
        candidate_normalized = input_transform(candidate)

        # Update expanded training data
        # Repeat for each task
        candidate_expanded = candidate_normalized.unsqueeze(1).repeat(1, num_tasks, 1)  # [1, num_tasks, D]
        tasks_candidate = task_indices.unsqueeze(0)  # [1, num_tasks]
        candidate_expanded = torch.cat([candidate_expanded, tasks_candidate.unsqueeze(-1)], dim=-1)  # [1, num_tasks, D+1]
        candidate_expanded = candidate_expanded.view(-1, candidate_expanded.shape[-1])  # [num_tasks, D+1]

        y_new_expanded = y_new.view(-1, 1)  # [num_tasks, 1]

        train_X_expanded = torch.cat([train_X_expanded, candidate_expanded], dim=0)
        train_Y_expanded = torch.cat([train_Y_expanded, y_new_expanded], dim=0)

        # Print progress
        printing(f"Iteration {iteration + 1}/{num_iterations}")
        printing(f"Candidate: {candidate}")
        printing(f"Objective values: {y_new}")

    # After optimization, process the results
    # Compute feasibility mask for final model predictions
    with torch.no_grad():
        posterior = model.posterior(train_X_expanded)
        mean = posterior.mean.view(-1, num_tasks)

    is_feasible = (c1(mean) >= 0) & (c2(mean) >= 0) & (c3(mean) >= 0) & (c4(mean) >= 0)
    is_feasible = is_feasible.all(dim=-1)

    # Get feasible train_Y and train_X
    feasible_Y = mean[is_feasible]
    feasible_X = train_X[is_feasible]

    # Extract the Pareto front
    pareto_mask = is_non_dominated(feasible_Y)
    pareto_front = feasible_Y[pareto_mask]
    pareto_points = feasible_X[pareto_mask]

    # Save Pareto front to CSV
    pareto_df = pd.DataFrame(
        torch.cat([pareto_points, pareto_front], dim=1).numpy(),
        columns=['sr', 'ht', 'cs', 'theta_deg', 'c-param', 'b-param', 'b_var']
    )
    pareto_df.to_csv(os.path.join(file_home_path, 'pareto_front.csv'), index=False)
