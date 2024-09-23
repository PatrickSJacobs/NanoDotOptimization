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
    df1 = pd.read_csv(main_work_dir + "ag-dot-angle-pretraining-unpruned.csv")

    parameters = df1[['sr', 'ht', 'cs', 'theta_deg']].values
    objectives = df1[['c-param', 'b-param', 'b_var']].values
    
    # Define kernel for GPR
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))

    # Initialize GPR models for each output dimension
    gpr_models = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10) for _ in range(3)]

    # Train a GPR model for each output dimension
    for i in range(3):
        gpr_models[i].fit(parameters, objectives[:, i])
        
    # Test data: new input of shape m x 4
    X_test = np.random.rand(10, 4)  # 10 samples with 4 features each

    # Predict for each output dimension
    y_pred = np.zeros((X_test.shape[0], 3))  # Store predictions for each of the 3 outputs
    y_std = np.zeros((X_test.shape[0], 3))   # Store standard deviations for each of the 3 outputs

    for i in range(3):
        y_pred[:, i], y_std[:, i] = gpr_models[i].predict(X_test, return_std=True)

    # Output predictions and uncertainties
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


