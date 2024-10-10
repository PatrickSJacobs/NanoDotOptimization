
from datetime import datetime
from time import sleep
import csv
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import sys
import traceback
import time
import statistics

logshift = lambda x: np.log(x + 1)

info_file = sys.argv[1]
#no_metal = True
#execution_dictionary = {}

info = [line.strip() for line in open(info_file, "r")]

progress_file = info[0]
air_data_path = info[1]
metal_data_path = info[2]
file_home_path = info[3]
file_work_path = info[4]
filename = info[5]
ticker_file = info[6]
air_raw_path = info[7]
metal_raw_path = info[8]
sr = info[9]
ht = info[10]
cs = info[11]
theta_deg = info[12] 

os.system("echo 'now for file input';%s" % ("\n"))  # Execute the simulation file
os.system("echo %s" % (info))  # Execute the simulation file

os.system("echo %s" % (air_data_path))  # Execute the simulation file
os.system("echo %s" % (metal_data_path))  # Execute the simulation file



def printing(string):
    file_printout = open(progress_file, 'r').readlines()
    lines = file_printout + [f"{str(string)}\n"]
    file_printer = open(progress_file, 'w')
    file_printer.writelines(lines)
    file_printer.close()
    print(string)
    pass



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

    if any(r > 1.0 for r in ys):
        return np.inf, np.inf, np.inf, np.inf
    
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

        weights = np.ceil(np.array(ys_fixed) / np.min(ys_fixed))
        weights = np.clip(weights, 0, 3000)  # Limit weights to prevent excessive memory usage

        L = np.repeat([xs[i] for i in index_list], weights.astype(int))

        maxi = statistics.mean(L) if len(L) > 0 else xs[ys.index(mam)]

    else:
        maxi = xs[ys.index(mam)]
        
    if maxi < 0.5:
        return np.inf, np.inf, np.inf, np.inf

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

    #return logshift(abs(b)), logshift(abs(c**2 * 10 - 10)), logshift(abs(b_var * 100)), logshift(abs(c_var * 100))
    return abs(b), abs(c**2 * 10 - 10), abs(b_var) * 100, abs(c_var) * 100

'''
success = 0

#(4) Extracting Data From optimization
max_time = (60*60)

time_count = 0
# Wait for data to be stable and ready for processing
while success == 0:
    try:
        if os.path.isfile(metal_data_path) and os.path.isfile(air_data_path):
            printing(f"files pass:{(air_data_path, metal_data_path)}")
            success = 1   
    except:
        print(metal_data_path)
        print(air_data_path)
    
    if time_count == max_time:
            raise Exception(f"files not existing: {(air_data_path, metal_data_path)}")

    time_count = time_count + 1
    time.sleep(1)
'''     

#b, c, b_var, c_var = 1E6, 1E6, 1E6, 1E6
b, c, b_var, c_var = np.inf, np.inf, np.inf, np.inf

# Check if data is good and data file exists, if not error
if os.path.isfile(metal_data_path) and os.path.isfile(air_data_path):
    df = None
    df0 = None
    try:
        df = pd.read_csv(metal_data_path, header=None)
        df0 = pd.read_csv(air_data_path, header=None)
    
        printing("success df")

        # Get wavelengths and reflectance data
        wvls = []
        R_meep = []
        
        for r, r0, wvl in zip(df[3], df0[3], df[1]):
            try:
                ration = np.abs(- float(r) / float(r0))
                R_meep += [ration]
                wvls += [float(wvl) * 0.32]
                
            except:
                pass

        #print(wvls)
        #print(R_meep)
        
        log_obj_exe = open(file_home_path + "calc_log_obj.csv", 'r').readlines()
        step_count = int(len(log_obj_exe))

        with open(file_work_path + "calc_log_data_step_%s_%s.csv" % (step_count, filename), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["wvl", "refl"])
            for (wvl, refl) in zip(wvls, R_meep):
                writer.writerow([wvl, refl])
            file.close()  
        printing("passed to obj")
        # (5) Sending Data Through Objective function

        b, c, b_var, c_var = obj_func_calc(wvls, R_meep)

        printing("came out of obj")

        #execution_dictionary[filename] = {"b": b, "c": c, "b_var": b_var, "c_var": c_var}
     
    except Exception as e:
        traceback.print_exc()
        
        ''' 
        print("An error occurred:")
        print(f"air_data_path: {air_data_path}")
        print(f"metal_data_path: {metal_data_path}")
        print(f"ticker_file: {ticker_file}")
        print(f"air_raw_path: {air_raw_path}")
        print(f"metal_raw_path: {metal_raw_path}")
        traceback.print_exc()
        os.system("scancel -u tg881088")
        sys.exit()
        '''

else:
    printing(f"({metal_data_path}): It isn't a file! Error in variables {[sr, ht, cs, theta_deg]} ")
    #raise ValueError(f"({metal_data_path}): It isn't a file!")

# (7) Logging of Current Data
with open(file_home_path + "calc_log_obj.csv", 'a') as file:
   
    writer = csv.writer(file)
    writer.writerow([filename, sr, ht, cs, theta_deg, b, c, b_var, c_var, datetime.now().strftime("%m_%d_%Y__%H_%M_%S"), step_count])
    file.close()
    

