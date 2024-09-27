#!/usr/bin/env python3
# coding: utf-8

# ## Constrained, Parallel, Multi-Objective BO in BoTorch with qNEHVI

# ### Import Statements

import os
import time
import warnings
import pandas as pd
import numpy as np
import torch

from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize

from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib

from botorch.exceptions import BadInitialCandidatesWarning  # Added import
from botorch import fit_gpytorch_mll

# ### Set dtype and device

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda:3" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# ### Define Constraint Functions (Relative to Objectives)

def constraint_function1(obj):
    """
    Constraint 1: f1 + f2 <= 10
    Since objectives are negated (obj = -f), the constraint becomes:
    -(obj1 + obj2) <= 10  => obj1 + obj2 >= -10
    To fit BoTorch's constraint <=0 format:
    -(obj1 + obj2 + 10) <=0
    """
    return -(obj[..., 0] + obj[..., 1] + 10)  # <=0 if satisfied

def constraint_function2(obj):
    """
    Constraint 2: f3 >= 5
    Adjusted for 3 objectives:
    Since obj3 = -f3:
    obj3 <= -5
    To fit BoTorch's constraint <=0 format:
    obj3 + 5 <=0
    """
    return obj[..., 2] + 5  # <=0 if satisfied

def constraint_function3(obj):
    """
    Constraint 3: f1 * f3 <= 20
    Adjusted for negated objectives:
    -(obj1) * -(obj3) <= 20  => obj1 * obj3 <=20
    To fit BoTorch's constraint <=0 format:
    obj1 * obj3 - 20 <=0
    """
    return obj[..., 0] * obj[..., 2] - 20  # <=0 if satisfied

def constraint_function4(obj):
    """
    Constraint 4: |f2| <=5
    Adjusted for 3 objectives:
    Since obj2 = -f2:
    | -obj2 | <=5  => |obj2| <=5
    To fit BoTorch's constraint <=0, define two constraints:
    obj2 - 5 <=0 and -obj2 -5 <=0
    """
    return obj[..., 1] - 5, -obj[..., 1] - 5  # Both <=0 if satisfied

def constraint_function5(obj):
    """
    Constraint 5: f1 <= 3
    Adjusted for 3 objectives:
    Since obj1 = -f1:
    -obj1 <=3  => obj1 >= -3
    To fit BoTorch's constraint <=0 format:
    -(obj1 + 3) <=0
    """
    return -(obj[..., 0] + 3)  # <=0 if satisfied

def constraint_function6(obj):
    """
    Constraint 6: f2 + f3 >=8
    Adjusted for negated objectives:
    -(obj2) - obj3 >=8  => obj2 + obj3 <= -8
    To fit BoTorch's constraint <=0 format:
    obj2 + obj3 +8 <=0
    """
    return obj[..., 1] + obj[..., 2] + 8  # <=0 if satisfied

def constraints(obj):
    """
    Evaluates all constraint functions at given objective values.

    Args:
        obj (torch.Tensor): Tensor of shape (..., M).

    Returns:
        torch.Tensor: Tensor of shape (..., num_constraints) with constraints <=0 if satisfied.
    """
    c1 = constraint_function1(obj)
    c2 = constraint_function2(obj)
    c3 = constraint_function3(obj)
    c4a, c4b = constraint_function4(obj)
    c5 = constraint_function5(obj)
    c6 = constraint_function6(obj)
    return torch.stack([c1, c2, c3, c4a, c4b, c5, c6], dim=-1)

# ### Problem Setup

# Define your custom problem class
class NanoDotProblem:
    def __init__(self, bounds, num_objectives, ref_point, **tkwargs):
        self.bounds = bounds
        self.num_objectives = num_objectives
        self.dim = bounds.shape[1]
        # Define your reference point (must be worse than any feasible objective value)
        self.ref_point = torch.tensor(ref_point, **tkwargs)
        # Set maximum hypervolume if known (optional)
        self.max_hv = None  # Replace with actual value if known

    def __call__(self, x):
        """
        Evaluates the objectives at x.

        Args:
            x (torch.Tensor): Tensor of shape (..., self.dim)

        Returns:
            torch.Tensor: Tensor of shape (..., self.num_objectives)
        """
        # Implement your objective functions here
        # Since we are minimizing, negate the objectives to convert to maximization
        # Replace the following with actual computations
        # Example placeholder objectives:
        # Replace these with actual computations based on your specific problem
        f1 = x[..., 0] ** 2 + x[..., 1]  # Example: f1 = x1^2 + x2
        f2 = torch.sin(x[..., 2]) + x[..., 3]  # Example: f2 = sin(x3) + x4
        f3 = torch.log1p(x[..., 0] * x[..., 3])  # Example: f3 = log(1 + x1*x4)

        obj1 = -f1  # Negate for minimization
        obj2 = -f2
        obj3 = -f3

        return torch.stack([obj1, obj2, obj3], dim=-1)

    def evaluate_constraints(self, obj):
        """
        Evaluates the constraints at given objective values.

        Args:
            obj (torch.Tensor): Tensor of shape (..., M)

        Returns:
            torch.Tensor: Tensor of shape (..., num_constraints) with constraints <=0 if satisfied.
        """
        return constraints(obj)

# Define the bounds for each parameter
# Replace lower_bound_param1, ..., upper_bound_param4 with actual values
lower_bound_param1 = 0.0
upper_bound_param1 = 10.0
lower_bound_param2 = 0.0
upper_bound_param2 = 10.0
lower_bound_param3 = 0.0
upper_bound_param3 = 10.0
lower_bound_param4 = 0.0
upper_bound_param4 = 10.0

bounds = torch.tensor([
    [lower_bound_param1, lower_bound_param2, lower_bound_param3, lower_bound_param4],
    [upper_bound_param1, upper_bound_param2, upper_bound_param3, upper_bound_param4]
], **tkwargs)

# ### Initial Data Loading

def load_initial_data_from_dataframe(df):
    """
    Loads initial training data from a pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing columns ['param1', 'param2', 'param3', 'param4', 'obj1', 'obj2', 'obj3'].

    Returns:
        tuple: (train_x, train_obj)
            - train_x (torch.Tensor): Tensor of shape (n_samples, d).
            - train_obj (torch.Tensor): Tensor of shape (n_samples, M).
    """
    train_x = torch.tensor(df[['param1', 'param2', 'param3', 'param4']].values, **tkwargs)
    train_obj = torch.tensor(df[['obj1', 'obj2', 'obj3']].values, **tkwargs)
    return train_x, train_obj

# Example usage:
# df_initial = pd.read_csv("initial_data.csv")  # Replace with your data source
# train_x, train_obj = load_initial_data_from_dataframe(df_initial)

# ### Model Initialization

def initialize_model(train_x, train_obj):
    """
    Initializes a multi-output GP model for the objectives.

    Args:
        train_x (torch.Tensor): Tensor of shape (n_samples, d).
        train_obj (torch.Tensor): Tensor of shape (n_samples, M).

    Returns:
        tuple: (mll, model)
            - mll (gpytorch.mlls.SumMarginalLogLikelihood): Marginal log likelihood.
            - model (ModelListGP): Multi-output GP model.
    """
    # Normalize inputs
    train_x_normalized = normalize(train_x, problem.bounds)
    # Define GP models for each objective
    models = []
    for i in range(train_obj.shape[-1]):
        models.append(
            SingleTaskGP(
                train_x_normalized, train_obj[..., i:i+1], outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def generate_initial_data(n):
    """
    Generates initial training data using Sobol sampling.

    Args:
        n (int): Number of initial samples to generate.

    Returns:
        tuple: (train_x, train_obj)
            - train_x (torch.Tensor): Tensor of shape (n, d).
            - train_obj (torch.Tensor): Tensor of shape (n, M).
    """
    # Generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj = problem(train_x)
    return train_x, train_obj

# ### Define the qNEHVI Acquisition Function Helper

def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler):
    """
    Optimizes the qNEHVI acquisition function and returns new candidate points and their evaluations.

    Args:
        model (ModelListGP): The surrogate GP model.
        train_x (torch.Tensor): Tensor of shape (n_samples, d).
        train_obj (torch.Tensor): Tensor of shape (n_samples, M).
        sampler (SobolQMCNormalSampler): Sampler for Monte Carlo integration.

    Returns:
        tuple: (new_x, new_obj)
            - new_x (torch.Tensor): Tensor of shape (BATCH_SIZE, d).
            - new_obj (torch.Tensor): Tensor of shape (BATCH_SIZE, M).
    """
    # Normalize training inputs
    train_x_norm = normalize(train_x, problem.bounds)
    
    # Define the qNEHVI acquisition function
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),
        X_baseline=train_x_norm,
        sampler=sampler,
        prune_baseline=True,
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(problem.num_objectives))),
        constraints=problem.evaluate_constraints,  # Pass the constraints directly
    )
    
    # Define bounds in normalized space
    standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
    standard_bounds[1] = 1
    
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
    
    # Unnormalize the candidates to original space
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    
    # Evaluate objectives at new candidates
    new_obj = problem(new_x)
    
    return new_x, new_obj

# ### Perform Bayesian Optimization Loop with qNEHVI Only

# Suppress specific warnings
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define BO parameters
N_BATCH = 20 if not SMOKE_TEST else 1
MC_SAMPLES = 128 if not SMOKE_TEST else 16
BATCH_SIZE = 2
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
verbose = True

# Initialize hypervolume calculator (to be set after reference point is determined)
hv = None  # Placeholder, will be initialized later

hvs_qnehvi, hvs_random = [], []

# ### Generate Initial Training Data and Compute Reference Point

# Define your reference points based on initial objective values
def compute_reference_point(train_obj, margin=0.05):
    """
    Computes the reference point based on initial objective values.

    Args:
        train_obj (torch.Tensor): Tensor of shape (n_samples, M).
        margin (float): Fractional margin to set the reference point slightly worse.

    Returns:
        list: Reference point values for each objective.
    """
    # Find the maximum observed objective values (since objectives are negated for minimization)
    max_obj = train_obj.max(dim=0).values  # Shape: (M,)
    
    # Apply margin to set the reference point slightly worse
    # Since we are maximizing obj, "worse" means lower values
    ref_point = (max_obj - margin * torch.abs(max_obj)).tolist()
    
    return ref_point

# Instantiate the problem with a temporary reference point
temporary_ref_point = [0.0, 0.0, 0.0]
problem = NanoDotProblem(bounds=bounds, num_objectives=3, ref_point=temporary_ref_point)

# Generate initial training data
initial_n = 2 * (4 + 1)  # d=4
train_x_initial, train_obj_initial = generate_initial_data(n=initial_n)
print(f"Initial training data shape: {train_x_initial.shape}, {train_obj_initial.shape}")

# Compute reference point based on initial objective values
ref_point = compute_reference_point(train_obj_initial, margin=0.05)
print(f"Computed reference point: {ref_point}")

# Re-instantiate the problem with the actual reference point
problem = NanoDotProblem(bounds=bounds, num_objectives=3, ref_point=ref_point)

# Initialize hypervolume calculator now that reference point is set
hv = Hypervolume(ref_point=problem.ref_point)

# Compute initial hypervolume for qNEHVI
def compute_hypervolume(train_x, train_obj, problem, hv):
    """
    Computes the hypervolume of the Pareto front.

    Args:
        train_x (torch.Tensor): Tensor of shape (n_samples, d).
        train_obj (torch.Tensor): Tensor of shape (n_samples, M).
        problem (NanoDotProblem): The optimization problem instance.
        hv (Hypervolume): Hypervolume calculator instance.

    Returns:
        float: Computed hypervolume.
    """
    # Identify feasible points (all constraints <=0)
    constraint_values = problem.evaluate_constraints(train_obj)
    is_feasible = (constraint_values <= 0).all(dim=-1)
    feasible_obj = train_obj[is_feasible]

    if feasible_obj.shape[0] == 0:
        return 0.0

    # Identify Pareto optimal points
    pareto_mask = is_non_dominated(feasible_obj)
    pareto_y = feasible_obj[pareto_mask]

    if pareto_y.shape[0] == 0:
        return 0.0

    # Compute hypervolume
    volume = hv.compute(pareto_y)
    return volume

initial_hv_qnehvi = compute_hypervolume(train_x_initial, train_obj_initial, problem, hv)
hvs_qnehvi.append(initial_hv_qnehvi)
print(f"Initial hypervolume for qNEHVI: {initial_hv_qnehvi}")

# Initialize random sampling baseline
train_x_random, train_obj_random = generate_initial_data(n=initial_n)
print(f"Initial random sampling data shape: {train_x_random.shape}, {train_obj_random.shape}")

# Compute initial hypervolume for random sampling
initial_hv_random = compute_hypervolume(train_x_random, train_obj_random, problem, hv)
hvs_random.append(initial_hv_random)
print(f"Initial hypervolume for random sampling: {initial_hv_random}")

# Initialize models
mll_qnehvi, model_qnehvi = initialize_model(
    train_x_initial, train_obj_initial
)

# Define the sampler
qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

# Run N_BATCH rounds of Bayesian Optimization
for iteration in range(1, N_BATCH + 1):
    t0 = time.monotonic()

    # Fit the qNEHVI model using fit_gpytorch_mll
    try:
        fit_gpytorch_mll(mll_qnehvi)
    except Exception as e:
        print(f"Model fitting failed at iteration {iteration}: {e}")
        break

    # Optimize acquisition function and get new observations
    try:
        new_x_qnehvi, new_obj_qnehvi = optimize_qnehvi_and_get_observation(
            model_qnehvi, train_x_initial, train_obj_initial, qnehvi_sampler
        )
    except Exception as e:
        print(f"Acquisition optimization failed at iteration {iteration}: {e}")
        break

    # Update training data for qNEHVI
    train_x_initial = torch.cat([train_x_initial, new_x_qnehvi])
    train_obj_initial = torch.cat([train_obj_initial, new_obj_qnehvi])

    # Random sampling for baseline
    new_x_random, new_obj_random = generate_initial_data(n=BATCH_SIZE)
    train_x_random = torch.cat([train_x_random, new_x_random])
    train_obj_random = torch.cat([train_obj_random, new_obj_random])

    # Compute hypervolume for qNEHVI and random
    volume_qnehvi = compute_hypervolume(train_x_initial, train_obj_initial, problem, hv)
    volume_random = compute_hypervolume(train_x_random, train_obj_random, problem, hv)

    hvs_qnehvi.append(volume_qnehvi)
    hvs_random.append(volume_random)

    # Reinitialize the qNEHVI model for the next iteration
    mll_qnehvi, model_qnehvi = initialize_model(
        train_x_initial, train_obj_initial
    )

    t1 = time.monotonic()

    if verbose:
        print(
            f"\nBatch {iteration:>2}: Hypervolume (random, qNEHVI) = "
            f"({volume_random:>4.2f}, {volume_qnehvi:>4.2f}), "
            f"time = {t1-t0:>4.2f} seconds.",
            end="",
        )
    else:
        print(".", end="")

# ### Extract and Print the Final Pareto Front

def extract_pareto_front(train_obj, problem):
    """
    Extracts the Pareto front from the training data.

    Args:
        train_obj (torch.Tensor): Tensor of shape (n_samples, M) representing objective values.
        problem (NanoDotProblem): The optimization problem instance.

    Returns:
        torch.Tensor: Tensor containing Pareto optimal objective values.
    """
    # Identify feasible points (all constraints <=0)
    constraint_values = problem.evaluate_constraints(train_obj)
    is_feasible = (constraint_values <= 0).all(dim=-1)
    feasible_obj = train_obj[is_feasible]

    if feasible_obj.shape[0] == 0:
        print("No feasible solutions found.")
        return torch.empty(0, train_obj.shape[-1])

    # Identify non-dominated points
    pareto_mask = is_non_dominated(feasible_obj)
    pareto_front = feasible_obj[pareto_mask]

    return pareto_front

def save_pareto_front(pareto_front, filename="pareto_front_qnehvi.csv"):
    """
    Saves the Pareto front to a CSV file.

    Args:
        pareto_front (torch.Tensor): Tensor containing Pareto optimal objective values.
        filename (str): Name of the file to save the Pareto front.
    """
    if pareto_front.numel() == 0:
        print("No Pareto front to save.")
        return

    # Re-negate objectives to original minimization scale
    pareto_front_original = -pareto_front  # Since obj = -f

    pareto_front_np = pareto_front_original.cpu().numpy()
    df_pareto = pd.DataFrame(pareto_front_np, columns=[f"Objective_{i+1}" for i in range(pareto_front_np.shape[-1])])
    df_pareto.to_csv(filename, index=False)
    print(f"Pareto front saved to {filename}.")

# Extract Pareto front for qNEHVI
pareto_front_qnehvi = extract_pareto_front(train_obj_initial, problem)

if pareto_front_qnehvi.numel() > 0:
    print("\nFinal Pareto Front for qNEHVI:")
    # Convert to NumPy for better readability (optional)
    pareto_front_qnehvi_np = -pareto_front_qnehvi.cpu().numpy()  # Re-negate to original minimization
    for idx, obj in enumerate(pareto_front_qnehvi_np, start=1):
        print(f"Pareto Point {idx}: {obj}")
    # Optionally, save to a file
    save_pareto_front(pareto_front_qnehvi)
else:
    print("\nNo feasible Pareto front found for qNEHVI.")

# ### Plot the Observations Colored by Iteration and Overlay Pareto Front

# Ensure the non-interactive backend is set
matplotlib.use('Agg')

fig, axes = plt.subplots(1, 2, figsize=(17, 5))
algos = ["Sobol", "qNEHVI"]
cm = plt.get_cmap("viridis")

batch_number_qnehvi = torch.cat(
    [
        torch.zeros(initial_n, device=tkwargs['device']),
        torch.arange(1, N_BATCH + 1, device=tkwargs['device']).repeat_interleave(BATCH_SIZE)
    ]
).cpu().numpy()

batch_number_random = torch.cat(
    [
        torch.zeros(initial_n, device=tkwargs['device']),
        torch.arange(1, N_BATCH + 1, device=tkwargs['device']).repeat_interleave(BATCH_SIZE)
    ]
).cpu().numpy()

for i, (train_obj, batch_num, algo) in enumerate([
    (train_obj_random, batch_number_random, "Sobol"),
    (train_obj_initial, batch_number_qnehvi, "qNEHVI")
]):
    # Re-negate objectives for plotting to original minimization scale
    train_obj_plot = -train_obj.cpu().numpy()
    
    sc = axes[i].scatter(
        train_obj_plot[:, 0],
        train_obj_plot[:, 1],
        c=batch_num,
        alpha=0.8,
        cmap=cm,
        label=f"{algo} Observations"
    )
    axes[i].set_title(f"{algo} Observations")
    axes[i].set_xlabel("Objective 1")
    axes[i].set_xlim(-15, 15)  # Adjust based on your objective ranges
    axes[i].set_ylim(-15, 15)
    if i == 0:
        axes[i].set_ylabel("Objective 2")
    
    # Overlay Pareto front if qNEHVI
    if algo == "qNEHVI" and pareto_front_qnehvi.numel() > 0:
        pareto_front_np = -pareto_front_qnehvi.cpu().numpy()  # Re-negate to original minimization
        axes[i].scatter(
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
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")

# Add legends
for ax in axes:
    ax.legend()

# Save the figure to the local directory
plt.savefig("observations_colored_by_iteration_with_pareto.png", bbox_inches='tight')
plt.close()
