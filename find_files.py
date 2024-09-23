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
    filter_strings = [filename_tag([0.08056, 0.07215, 0.5801, 0.0])]
    
    filtered_files = collect_calc_log_files(main_work_dir, filter_strings)
    
    for file in filtered_files:
        print(file)
