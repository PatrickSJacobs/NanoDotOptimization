
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import re

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

main_home_dir = "/home1/08809/tg881088/" # Home directory for optimization
main_work_dir = "/work2/08809/tg881088/" # Home directory for optimization

collection_file = open(main_home_dir + "ag-dot-angle-pretraining.csv", 'w', newline='')
writer = csv.writer(collection_file)
writer.writerow(["path", "sr", "ht", "cs", "theta_deg", "b-param", "c-param", "b_var", "c_var", "count"])

# Print the collected calc_log files
count = 0

for file_set in collect_calc_log_files(main_work_dir):
    
    path = file_set[0]
    date = file_set[1]
    
    count += 1
    
    scaling_factor = 10000
    
    if date < 20231127:
        scaling_factor = 1000

    
    sr = float(path.split('_sr_')[1].split('nm_')[0].replace("_", ".")) / scaling_factor
    ht =  float(path.split('_ht_')[1].split('nm_')[0].replace("_", ".")) / scaling_factor
    cs = float(path.split('_cs_')[1].split('nm_')[0].replace("_", ".")) / scaling_factor
    theta_deg = float(path.split('_deg_')[1].split('.csv')[0].replace("_", ".")) % 360
        
    data = pd.read_csv(path)
    wvls = data["wvl"].tolist()
    R_meep = data["refl"].tolist()

    b, c, b_var, c_var = obj_func_calc(wvls, R_meep)
    
    writer.writerow([path, sr, ht, cs, theta_deg, b, c, b_var, c_var, count])

collection_file.close()  