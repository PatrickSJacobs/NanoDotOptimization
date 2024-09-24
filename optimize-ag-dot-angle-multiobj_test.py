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

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.optim import optimize_acqf

from gpytorch.mlls import SumMarginalLogLikelihood

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
    #return 5 - samples[..., 0]  # c-param <= 5
    return samples[..., 0]  # c-param <= 5

def c2(samples):
    #return samples[..., 1] - 1  # b-param >= 1
    return samples[..., 1]  # b-param >= 1

def c3(samples):
    #return 50 - samples[..., 1]  # b-param <= 50
    return samples[..., 1]  # b-param <= 50

def c4(samples):
    #return 10 - samples[..., 2]  # b_var <= 10
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

# Custom Posterior class to combine posterior samples
from botorch.posteriors import Posterior
from torch.distributions import Normal

class CombinedPosterior(Posterior):
    def __init__(self, posteriors):
        self.posteriors = posteriors
        self.device = posteriors[0].device
        self.dtype = posteriors[0].dtype

    @property
    def mean(self):
        means = [p.mean for p in self.posteriors]
        return torch.cat(means, dim=-1)

    @property
    def variance(self):
        variances = [p.variance for p in self.posteriors]
        return torch.cat(variances, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        samples = [p.rsample(sample_shape) for p in self.posteriors]
        return torch.cat(samples, dim=-1)

# Custom ModelListGP wrapper to combine posteriors
class CustomModelListGP(ModelListGP):
    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        posteriors = [model.posterior(X, observation_noise=observation_noise, **kwargs) for model in self.models]
        combined_posterior = CombinedPosterior(posteriors)
        return combined_posterior

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

    # Inputs (include 'cs' since it's now a variable)
    train_X = torch.tensor(df[['sr', 'ht', 'cs', 'theta_deg']].values, dtype=torch.double)
    print(len(train_X))

    # Outputs
    train_Y = torch.tensor(df[['c-param', 'b-param', 'b_var']].values, dtype=torch.double)
    print(len(train_Y))

    # Bounds (include 'cs' bounds)
    bounds = torch.tensor([
        [0.005, 0.05, 0.025, 0.0],   # Lower bounds for sr, ht, cs, theta_deg
        [0.125, 0.1, 0.25, 90.0]     # Upper bounds for sr, ht, cs, theta_deg
    ], dtype=torch.double)

    print("not normalized")
    # Normalize the training inputs
    train_X_normalized = (train_X - bounds[0]) / (bounds[1] - bounds[0])
    print("normalized")

    num_iterations = 4  # Number of optimization iterations

    # Initialize model
    def initialize_model(train_X, train_Y):
        models = []
        for i in range(train_Y.shape[-1]):
            model = SingleTaskGP(
                train_X,
                train_Y[:, i:i+1],
                outcome_transform=Standardize(m=1),
            )
            models.append(model)
        model = CustomModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    mll, model = initialize_model(train_X_normalized, train_Y)

    for iteration in range(num_iterations):
        # Fit the model
        fit_gpytorch_mll(mll)
        print("Model fitted")

        
        # Compute feasibility mask using raw outputs
        is_feasible = (c1(train_Y) >= 0) & (c2(train_Y) >= 0) & (c3(train_Y) >= 0) & (c4(train_Y) >= 0)
        is_feasible = is_feasible.all(dim=-1)

        if is_feasible.sum() == 0:
            printing("No feasible observations found.")
            break
        
        
        feasible_Y = train_Y[is_feasible]
        feasible_X = train_X_normalized[is_feasible]

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
            cache_root=False,  # Set to False when using custom models
        )

        # Optimize the acquisition function to get the next candidate
        candidate_normalized, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([
                torch.zeros(train_X_normalized.shape[-1]),
                torch.ones(train_X_normalized.shape[-1]),
            ]),
            q=1,
            num_restarts=5,
            raw_samples=20,  # For initialization
        )

        # Denormalize the candidate
        candidate = candidate_normalized * (bounds[1] - bounds[0]) + bounds[0]

        # Evaluate the candidate
        y_new = evaluate_candidate(candidate)

        # Update training data
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y_new], dim=0)

        # Normalize the new candidate
        candidate_normalized = (candidate - bounds[0]) / (bounds[1] - bounds[0])
        train_X_normalized = torch.cat([train_X_normalized, candidate_normalized], dim=0)

        # Reinitialize the model with the updated data
        mll, model = initialize_model(train_X_normalized, train_Y)

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
        columns=['sr', 'ht', 'cs', 'theta_deg', 'c-param', 'b-param', 'b_var']
    )
    pareto_df.to_csv(os.path.join(file_home_path, 'pareto_front.csv'), index=False)
