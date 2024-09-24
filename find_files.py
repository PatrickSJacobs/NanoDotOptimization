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
    
    '''
    # Array of strings to filter files by
    filter_strings = [filename_tag(i) for i in [
   [0.05103, 0.08608, 0.49075, 0.0]
   [0.0669, 0.0759, 0.2662, 0.0],
    [0.06791, 0.05, 0.25, 0.0],
    [0.07675, 0.05137, 0.1771, 0.0],
    
    [0.08373, 0.05565, 0.13783, 0.0],
    [0.0539, 0.1021, 0.2922, 0.0],
    [0.0531, 0.0529, 0.2938, 0.0],
    [0.07541, 0.05, 0.25, 0.0],
    [0.08129, 0.05029, 0.1451, 54.9],
    [0.059, 0.0653, 0.282, 0.0],
    [0.05787, 0.05023, 0.25, 0.0],
    [0.08083, 0.058, 0.25, 0.0],
    [0.0434, 0.1074, 0.3132, 0.0],
    [0.07162, 0.05365, 0.19235, 0.0],
    [0.04928, 0.06514, 0.54141, 0.0],
    [0.05817, 0.05041, 0.52902, 0.0],
    [0.07263, 0.05429, 0.25, 0.0],
    [0.0645, 0.0563, 0.271, 0.0],
    [0.0532, 0.0853, 0.2936, 0.0],
    [0.07106, 0.05, 0.24719, 0.0],
    [0.08065, 0.05247, 0.20044, 0.0],
    [0.08323, 0.05668, 0.24958, 0.0],
    [0.0529, 0.0651, 0.2942, 0.0],
    [0.0447, 0.1236, 0.3106, 0.0],
    [0.0617, 0.0559, 0.2766, 0.0],
    [0.05667, 0.0538, 0.24976, 0.0],
    [0.06582, 0.05717, 0.25, 0.0],
    [0.04927, 0.055, 0.24719, 0.0],
    [0.077, 0.0579, 0.246, 0.0],
    [0.07216, 0.05, 0.12685, 0.0],
    [0.08314, 0.05, 0.21324, 0.0],
    [0.0796, 0.0558, 0.2408, 0.0]# Solution 6 (repeated)
]
]'''
    
    # Array of strings to filter files by
    filter_strings = [filename_tag(i) for i in [
   [0.05103, 0.08608, 0.49075, 0.0]# Solution 6 (repeated)
]
]
       
       
    filtered_files = collect_calc_log_files(main_work_dir, filter_strings)
    
    for file in filtered_files:
        print(file)
