import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import pyplot
import statistics
from scipy.signal import find_peaks
from pathlib import Path

def list_files_in_directory(directory):
    # Create a Path object for the directory
    path = Path(directory)
    
    # List all files in the directory
    return [str(file) for file in path.iterdir() if file.is_file()]

directory_path = '/Users/calaeuscaelum/Documents/Development/Tang_Project/NanoDotOptimization/data'
files = list_files_in_directory(directory_path)

for file in files:
    # define the true objective function
    data = pd.read_csv(file)

    K = 2

    xs = data["wvl"].tolist()
    ys = data["refl"].tolist()

    xs = xs[: len(xs) - K]
    ys = ys[: len(ys) - K]

    plt.plot(xs, ys, color = 'grey' )

    plt.xlabel('wavelength')
    plt.ylabel('reflectance')

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

    #tsr = sum(x * i for i, x in enumerate(L, 1)) / len(L)

    #print(tsr)

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
    #print((b, c - 1000))

    # calculate the output for the range
    y_line = objective(xs, b, c)
    # create a line plot for the mapping function
    u = [xs[ind[i]] for i in peaks]
    p = [ys[ind[i]] for i in peaks]
    #u = [xs[i] for i in ind]
    #p = [ys[i] for i in ind]
    plt.plot(u, p, "x")
    plt.plot(xs, y_line, '--', color='red')
    plt.title(f"{file.split('/')[-1]}")
    plt.suptitle(f"b = {b:.2f}, c = {c:.2f}, b_var = {b_var:.2f}, c_var = {c_var:.2f}")
# Add the second subtitle using fig.text()
    pyplot.show()