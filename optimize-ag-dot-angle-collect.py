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
from scipy.stats import zscore
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
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

    ind = [i for i in range(min(ind), max(ind) + 1)]
    maxis = [ys[i] for i in ind]

    neg_ys_prime = [element + mam for element in [(-1) * y for y in maxis]]
    neg_ys = [element + mam for element in [(-1) * y for y in ys]]

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

        # Use a cumulative sum instead of creating a large list
        weighted_sum = sum(wvl * freq for wvl, freq in zip([xs[i] for i in index_list], np.ceil(np.array(ys_fixed) / np.min(ys_fixed))))
        total_freq = sum(np.ceil(np.array(ys_fixed) / np.min(ys_fixed)))
        
        if total_freq > 0:
            maxi = weighted_sum / total_freq
        else:
            maxi = xs[ys.index(mam)]

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

def prune_dataset(df, m, perform_cleaning=True, verbose=True):
    """
    Prunes the dataset to reduce it to m statistically significant points.

    Parameters:
    - df (pd.DataFrame): The original dataset.
    - m (int): The desired number of points after pruning.
    - perform_cleaning (bool): Whether to perform preliminary cleaning steps.
    - verbose (bool): Whether to print pruning steps' information.

    Returns:
    - pd.DataFrame: The pruned dataset containing m points.
    """
    if perform_cleaning:
        # Step 1: Remove Exact Duplicates
        df_dedup = df.drop_duplicates(subset=["sr", "ht", "cs", "theta_deg", 
                                             "b-param", "c-param", "b_var", "c_var"])
        if verbose:
            print(f"After removing exact duplicates: {len(df_dedup)} records")

        # Early Exit if Dataset Too Small
        if len(df_dedup) < m:
            if verbose:
                print(f"Dataset size {len(df_dedup)} is less than desired m={m}. Skipping further pruning.")
            return df_dedup.copy()

        # Step 2: Normalize Features
        features = ["sr", "ht", "cs", "theta_deg", "b-param", "c-param", "b_var", "c_var"]
        scaler = StandardScaler()
        df_normalized = df_dedup.copy()
        df_normalized[features] = scaler.fit_transform(df_dedup[features])
        if verbose:
            print("Features normalized.")

        # Step 3: Remove Near-Duplicates Using Clustering (DBSCAN)
        dbscan_eps = 0.5  # Initial value, can be adjusted
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=1, metric='euclidean')
        clusters = dbscan.fit_predict(df_normalized[features])
        df_normalized['cluster'] = clusters
        df_near_dedup = df_normalized.groupby('cluster').first().reset_index(drop=True)
        df_pruned = df_dedup.iloc[df_near_dedup.index]
        if verbose:
            print(f"After removing near-duplicates with DBSCAN (eps={dbscan_eps}): {len(df_pruned)} records")

        # Check if dataset is still >= m
        if len(df_pruned) < m:
            if verbose:
                print(f"Dataset size {len(df_pruned)} after near-duplicate removal is less than m={m}. Adjusting DBSCAN parameters.")
            # Adjust DBSCAN to be less aggressive
            dbscan_eps = dbscan_eps * 1.5  # Increase eps to cluster fewer points
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=1, metric='euclidean')
            clusters = dbscan.fit_predict(df_normalized[features])
            df_normalized['cluster'] = clusters
            df_near_dedup = df_normalized.groupby('cluster').first().reset_index(drop=True)
            df_pruned = df_dedup.iloc[df_near_dedup.index]
            if verbose:
                print(f"After adjusting DBSCAN (eps={dbscan_eps}): {len(df_pruned)} records")
            
            # If still too small, skip near-duplicate removal
            if len(df_pruned) < m:
                if verbose:
                    print(f"Dataset size {len(df_pruned)} is still less than m={m}. Skipping near-duplicate removal.")
                df_pruned = df_dedup.copy()

        # Step 4: Filter Based on Feature Variance
        z_scores = zscore(df_pruned[features])
        abs_z_scores = abs(z_scores)
        mask = (abs_z_scores > 1).any(axis=1)
        df_variance_filtered = df_pruned[mask]
        if verbose:
            print(f"After variance filtering (z-score > 1): {len(df_variance_filtered)} records")

        # Check if dataset is still >= m
        if len(df_variance_filtered) < m:
            if verbose:
                print(f"Dataset size {len(df_variance_filtered)} after variance filtering is less than m={m}. Adjusting variance threshold.")
            # Relax variance threshold
            mask = (abs_z_scores > 0.5).any(axis=1)
            df_variance_filtered = df_pruned[mask]
            if verbose:
                print(f"After relaxing variance filtering (z-score > 0.5): {len(df_variance_filtered)} records")
            
            # If still too small, skip variance filtering
            if len(df_variance_filtered) < m:
                if verbose:
                    print(f"Dataset size {len(df_variance_filtered)} is still less than m={m}. Skipping variance filtering.")
                df_variance_filtered = df_pruned.copy()

        # Step 5: Apply Outlier Detection
        contamination = 0.05  # Initial value, can be adjusted
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_preds = iso_forest.fit_predict(df_variance_filtered[features])
        df_no_outliers = df_variance_filtered[outlier_preds == 1]
        if verbose:
            print(f"After outlier removal (contamination={contamination}): {len(df_no_outliers)} records")

        # Check if dataset is still >= m
        if len(df_no_outliers) < m:
            if verbose:
                print(f"Dataset size {len(df_no_outliers)} after outlier removal is less than m={m}. Adjusting contamination rate.")
            # Decrease contamination to remove fewer outliers
            contamination = contamination * 0.5  # Reduce contamination rate
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_preds = iso_forest.fit_predict(df_variance_filtered[features])
            df_no_outliers = df_variance_filtered[outlier_preds == 1]
            if verbose:
                print(f"After adjusting outlier removal (contamination={contamination}): {len(df_no_outliers)} records")
            
            # If still too small, skip outlier removal
            if len(df_no_outliers) < m:
                if verbose:
                    print(f"Dataset size {len(df_no_outliers)} is still less than m={m}. Skipping outlier removal.")
                df_no_outliers = df_variance_filtered.copy()

        # Step 6: Dimensionality Reduction (PCA)
        pca = PCA(n_components=0.95, random_state=42)
        df_pca = pca.fit_transform(df_no_outliers[features])
        if verbose:
            print(f"Reduced to {df_pca.shape[1]} principal components (95% variance retained)")

        # Step 7: Final Clustering to Reduce to m Points
        if m >= len(df_no_outliers):
            if verbose:
                print(f"Desired number of points m={m} is greater than or equal to the dataset size {len(df_no_outliers)}. Returning the entire dataset.")
            return df_no_outliers.copy()

        kmeans = KMeans(n_clusters=m, random_state=42)
        clusters_final = kmeans.fit_predict(df_pca)
        df_no_outliers['final_cluster'] = clusters_final

        # Select the closest point to each cluster centroid
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, df_pca)
        df_final = df_no_outliers.iloc[closest].reset_index(drop=True)
        if verbose:
            print(f"After clustering to {m} points: {len(df_final)} records")

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
            if any(not math.isfinite(x) for x in [sr, ht, cs, theta_deg, b, c, b_var, c_var, count]):
                continue

            print(type(dataset_df))
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

    num_points = 1000
    # Prune the dataset
    df_final = prune_dataset(dataset_df, num_points)

    # Save the pruned dataset
    pruned_training_file = main_work_dir + "ag-dot-angle-pretraining.csv"
    df_final.to_csv(pruned_training_file, index=False)

    print(f"Final pruned dataset contains {len(df_final)} records.")

if __name__ == "__main__":
    main()
