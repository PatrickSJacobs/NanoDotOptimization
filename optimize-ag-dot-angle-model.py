# Standard Library Imports
import csv
import math
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
import statistics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt


if __name__ == "__main__":
    main_work_dir = "/Users/calaeuscaelum/Documents/Development/Tang_Project/NanoDotOptimization/data/"  # Home directory for optimization
    df1 = pd.read_csv(main_work_dir + "ag-dot-angle-pretraining.csv")

    parameters = df1[['sr', 'ht', 'cs', 'theta_deg']].values
    objectives = df1[['c-param', 'b-param', 'b_var']].values
    
    # Define kernel for GPR
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
    
    print("starting models")
    # Initialize GPR models for each output dimension
    gpr_models = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100) for _ in range(3)]
    print("made models")

    print("fitting models")
    # Train a GPR model for each output dimension
    for i in range(3):
        print(f"fitting model {i+1}")
        gpr_models[i].fit(parameters, objectives[:, i])
        print(f"finished fitting model {i+1}")

        
    # Test data: new input of shape m x 4
    X_test = np.array([
    [0.01265, 0.08967, 0.52237, 0.00000],  # Solution 29
    [0.0669, 0.0759, 0.2662, 0.00000],     # Solution 26
    [0.08129, 0.05029, 0.1451, 54.90000],  # Solution 6
    [0.0595, 0.0631, 0.281, 0.00000],      # Solution 12
    [0.07858, 0.05388, 0.25, 0.00000],     # Solution 2
    [0.0538, 0.1008, 0.2924, 0.00000],     # Solution 5
    [0.0348, 0.1235, 0.3304, 0.00000],     # Solution 4
    [0.00573, 0.07468, 0.52564, 0.00000],  # Solution 13
    [0.05667, 0.0538, 0.24976, 0.00000],   # Solution 18
    [0.07568, 0.05, 0.16359, 0.00000],     # Solution 20
    [0.0669, 0.0759, 0.2662, 0.00000],     # Solution 26 (repeated)
    [0.01265, 0.08967, 0.52237, 0.00000],  # Solution 29 (repeated)
    [0.11868, 0.05, 0.18418, 0.00000],     # Solution 30
    [0.09467, 0.05944, 0.22999, 0.00000],  # Solution 31
    [0.08129, 0.05029, 0.1451, 54.90000],  # Solution 6 (repeated)
])

    # Predict for each output dimension
    y_pred = np.zeros((X_test.shape[0], 3))  # Store predictions for each of the 3 outputs
    y_std = np.zeros((X_test.shape[0], 3))   # Store standard deviations for each of the 3 outputs

    for i in range(3):
        y_pred[:, i], y_std[:, i] = gpr_models[i].predict(X_test, return_std=True)

    # Output predictions and uncertainties
    print(X_test)
    print("Predicted values:\n", y_pred)
    print("\nPrediction uncertainties (std dev):\n", y_std)

    # For visualization, let's assume we're working with the first output dimension only
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(X_test.shape[0]), y_pred[:, 0], 'b-', label='Predicted Output 1')
    plt.fill_between(np.arange(X_test.shape[0]),
                    y_pred[:, 0] - 2 * y_std[:, 0], 
                    y_pred[:, 0] + 2 * y_std[:, 0],
                    color='lightblue', alpha=0.5, label='Confidence Interval (±2 std)')
    plt.title('GPR Prediction for Output 1 with Confidence Interval')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Output Value')
    plt.legend()
    plt.show()


