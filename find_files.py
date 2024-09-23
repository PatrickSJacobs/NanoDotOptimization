from pathlib import Path

def filename_tag(x: [float]):
    sr = x[0]
    ht = x[1]
    cs = x[2]
    theta_deg = x[3]
    display_theta_deg = str(round(theta_deg if theta_deg > 0 else theta_deg + 360.0,
                                  1)).replace(".", "_")  # angle to be used

    filename = "_sr_%s_ht_%s_cs_%s_theta_deg_%s" % (
                                                      str(round(sr * 10000, 1)).replace(".", "_") + "nm",
                                                      str(round(ht * 10000, 1)).replace(".", "_") + "nm",
                                                      str(round(cs * 10000, 1)).replace(".", "_") + "nm",
                                                      display_theta_deg,
                                                      )  # filename to be used
    return filename

def collect_calc_log_files(base_directory, filter_strings):
    """
    Collects files with 'calc_log' in their name and checks if the filename contains
    any of the strings in the filter_strings array.
    """
    base_dir = Path(base_directory)
    
    # Collect all directories with "opt_" in their name
    opt_dirs = [dir for dir in base_dir.glob('opt_*') if dir.is_dir()]
    
    calc_log_files = []
    
    # Loop through each "opt_" directory
    for opt_dir in opt_dirs:
        # Find files containing "calc_log" within each "opt_" directory
        calc_files = list(opt_dir.glob('**/*calc_log*'))
        
        # Filter files based on the strings in the filter_strings array
        for file in calc_files:
            # Check if any of the filter strings are in the file name
            if any(filter_str in file.name for filter_str in filter_strings):
                calc_log_files.append(str(file))

    return calc_log_files

if __name__ == "__main__":
    main_work_dir = "/work2/08809/tg881088/"  # Home directory for optimization
    
    # Array of strings to filter files by
    filter_strings = [filename_tag(i) for i in [
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
]
]
       
       
    filtered_files = collect_calc_log_files(main_work_dir, filter_strings)
    
    for file in filtered_files:
        print(file)
