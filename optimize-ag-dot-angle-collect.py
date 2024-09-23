# Standard Library Imports
import csv
import math
import os
import re
from pathlib import Path

# Third-Party Imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
# [Your other code goes here]

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

    ind = [i for i in range(min(ind), max(ind)+1)]
    maxis = [ys[i] for i in ind]

    neg_ys_prime = [element + mam for element in [(-1)*y for y in maxis]]
    neg_ys = [element + mam for element in [(-1)*y for y in ys]]

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


def date_to_scalar(year, month, day):
    """Convert date components to a scalar."""
    return year * 10000 + month * 100 + day

def extract_date_from_folder(folder_name):
    """Extracts the date from folder name in the form opt_MM_DD_YYYY."""
    # Match pattern 'opt_MM_DD_YYYY'
    match = re.search(r'opt_(\d{2})_(\d{2})_(\d{4})__', folder_name)
    if match:
        month, day, year = map(int, match.groups())  # Extract as integers
        return date_to_scalar(year, month, day)
    return None  # Return None if no date found

def collect_calc_log_files(base_directory):
    """Collects files with 'calc_log' in their name, along with a date scalar from the folder name."""
    base_dir = Path(base_directory)
    
    # Collect all directories with "opt_" in their name
    opt_dirs = [dir for dir in base_dir.glob('opt_*') if dir.is_dir()]
    
    calc_log_files = []
    
    # Loop through each "opt_" directory
    for opt_dir in opt_dirs:
        # Extract date scalar from the folder name
        date_scalar = extract_date_from_folder(opt_dir.name)
        
        if date_scalar is not None:  # If a valid date was extracted
            # Find files containing "calc_log" within each "opt_" directory
            calc_files = list(opt_dir.glob('**/*calc_log*'))
            
            # Add each file and its corresponding date scalar to the list
            for file in calc_files:
                calc_log_files.append([str(file), date_scalar])
    
    return calc_log_files

# [Include all your existing function definitions here: obj_func_calc, date_to_scalar, extract_date_from_folder, collect_calc_log_files]


def prune_dataset(df, m, thresholds=None, features=None, random_state=42, verbose=True):
    """
    Prunes the dataset by first removing rows with 'b-param', 'c-param', 'b_var',
    or 'c_var' exceeding specified thresholds, then reduces the dataset to m
    representative points using KMeans clustering.
    
    Parameters:
    - df (pd.DataFrame): The original dataset.
    - m (int): The desired number of points after pruning.
    - thresholds (dict, optional): Dictionary specifying the maximum allowable values
                                   for 'b-param', 'c-param', 'b_var', and 'c_var'.
                                   Example: {'b-param': 10, 'c-param': 20, 'b_var': 5, 'c_var': 5}
                                   If None, no initial filtering is applied.
    - features (list of str, optional): List of feature column names to use for clustering.
                                        If None, all numeric columns are used.
    - random_state (int, optional): Seed for reproducibility. Default is 42.
    - verbose (bool, optional): If True, prints information about pruning steps.
    
    Returns:
    - pd.DataFrame: The pruned dataset containing m representative points.
    """
    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    
    if not isinstance(m, int) or m <= 0:
        raise ValueError("Parameter m must be a positive integer.")
    
    # Step 1: Initial Filtering Based on Thresholds
    if thresholds:
        # Ensure all specified columns are present in the DataFrame
        missing_cols = [col for col in thresholds.keys() if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following threshold columns are not in the DataFrame: {missing_cols}")
        
        # Apply filtering: retain rows where all specified columns are <= thresholds
        condition = np.ones(len(df), dtype=bool)
        for col, thresh in thresholds.items():
            condition &= df[col] <= thresh
            if verbose:
                print(f"Applying threshold: {col} <= {thresh}")
        
        df_filtered = df[condition].copy()
        if verbose:
            print(f"After initial filtering based on thresholds: {len(df_filtered)} records retained.")
    else:
        df_filtered = df.copy()
        if verbose:
            print("No initial filtering based on thresholds applied.")
    
    # Check if there are enough data points after filtering
    if len(df_filtered) < m:
        if verbose:
            print(f"Warning: After filtering, the dataset contains {len(df_filtered)} records, which is less than the desired m={m}.")
            print("Proceeding with the available data without further pruning.")
        return df_filtered.copy()
    
    # Step 2: Feature Selection for Clustering
    if features is None:
        # Use all numeric columns if features not specified
        features = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        if not features:
            raise ValueError("No numeric columns available for clustering.")
        if verbose:
            print(f"Selected features for clustering (all numeric columns): {features}")
    else:
        # Ensure specified features exist in the DataFrame
        missing_features = [feat for feat in features if feat not in df_filtered.columns]
        if missing_features:
            raise ValueError(f"The following features are not in the DataFrame: {missing_features}")
        if verbose:
            print(f"Selected features for clustering: {features}")
    
    # Extract the feature matrix
    X = df_filtered[features].values
    
    # Handle missing values by imputing or removing
    if np.isnan(X).any():
        raise ValueError("Feature matrix contains NaN values. Please handle missing data before clustering.")
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if verbose:
        print("Features standardized using StandardScaler.")
    
    # Step 3: Check if m is feasible
    if m >= len(df_filtered):
        if verbose:
            print(f"Desired number of points m={m} is greater than or equal to the dataset size {len(df_filtered)}.")
            print("Returning the filtered dataset without further pruning.")
        return df_filtered.copy()
    
    # Step 4: Initialize and Fit KMeans
    kmeans = KMeans(n_clusters=m, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)
    if verbose:
        print(f"KMeans clustering performed with n_clusters={m}.")
    
    # Step 5: Find the Closest Data Point to Each Cluster Centroid
    centers = kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(centers, X_scaled)
    
    # Step 6: Select the Representative Points
    df_final = df_filtered.iloc[closest].reset_index(drop=True)
    if verbose:
        print(f"Pruned dataset contains {len(df_final)} records after KMeans clustering to {m} points.")
    
    return df_final

def main():
    main_home_dir = "/home1/08809/tg881088/"  # Home directory for optimization
    main_work_dir = "/work2/08809/tg881088/"  # Home directory for optimization

    training_file = main_work_dir + "ag-dot-angle-pretraining-unpruned.csv"

    # Initialize an empty DataFrame
    columns = ["path", "sr", "ht", "cs", "theta_deg", "b-param", "c-param", "b_var", "c_var", "count"]
    dataset_df = pd.DataFrame(columns=columns)
    
    count = 0
    
    for file_set in collect_calc_log_files(main_work_dir):
        path, date = file_set
        count += 1
        scaling_factor = 10000 if date >= 20231127 else 1000

        try:
            # Parse parameters from the file path
            sr = float(path.split('_sr_')[1].split('nm_')[0].replace("_", ".")) / scaling_factor
            ht = float(path.split('_ht_')[1].split('nm_')[0].replace("_", ".")) / scaling_factor
            cs = 0.4 - 2 * sr

            try:
                cs_split = path.split('_cs_')[1]
                cs = float(cs_split.split('nm_')[0].replace("_", ".")) / scaling_factor
            except IndexError:
                pass  # cs remains as 0.4 - 2 * sr if '_cs_' not found

            theta_deg = float(path.split('_deg_')[1].split('.csv')[0].replace("_", ".")) % 360

            # Read reflectance data
            data = pd.read_csv(path)
            wvls = data["wvl"].tolist()
            R_meep = data["refl"].tolist()

            # Calculate parameters
            b, c, b_var, c_var = obj_func_calc(wvls, R_meep)

            # Check for finite values
            if any(not math.isfinite(x) or x < 0 for x in [sr, ht, cs, theta_deg, b, c, b_var, c_var, count]):
                continue

            # Append the row to the DataFrame
            
            dataset_df =  pd.concat([dataset_df, 
                                     pd.DataFrame([{
                "path": path,
                "sr": sr,
                "ht": ht,
                "cs": cs,
                "theta_deg": theta_deg,
                "b-param": b,
                "c-param": c,
                "b_var": b_var,
                "c_var": c_var,
                "count": count
            }])], ignore_index=True)
        except Exception as e:
            print(f"Error processing file {path}: {e}")
            continue

    # Save the collected data to CSV
    dataset_df.to_csv(training_file, index=False)
    print(f"Collected dataset contains {len(dataset_df)} records before pruning.")

    num_points = 3000
    # Prune the dataset
    df_final = prune_dataset(
        dataset_df, 
        num_points, 
        { "c-param": 10, "b_var": 10, }
        )

    # Save the pruned dataset
    pruned_training_file = main_work_dir + "ag-dot-angle-pretraining.csv"
    df_final.to_csv(pruned_training_file, index=False)

    print(f"Final pruned dataset contains {len(df_final)} records.")

if __name__ == "__main__":
    main()
