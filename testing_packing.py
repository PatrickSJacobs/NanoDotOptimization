import math


def max_circle_radius(w, l):
    # Function to calculate the number of circles fitting in a given dimension
    def fit_count(size, radius):
        # Distance between centers in the hexagonal packing
        dist = math.sqrt(3) * radius
        return int(size / dist)

    # Initialize binary search for the maximum radius
    low, high = 0, min(w, l) / 2  # Reducing initial high estimate to a more realistic value
    best_radius = 0  # Store the best radius that allows fitting at least three circles

    while high - low > 1e-6:  # Precision of the radius
        mid = (low + high) / 2
        num_across = fit_count(w, mid)
        num_along = fit_count(l, mid)

        # Calculate the effective number of rows and columns considering hexagonal offset
        num_along_adjusted = num_along + (num_across - 1) * (num_along // 2 if num_along > 1 else 0)

        # Calculate the total number of circles
        total_circles = num_across * num_along_adjusted

        # Check if the circles fit within the rectangle and there are at least three circles
        if total_circles >= 3:
            best_radius = mid  # Update best found radius
            low = mid  # Increase radius
        else:
            high = mid  # Decrease radius

    return best_radius, total_circles


# Example usage
w = 10  # Width of the rectangle
l = 40  # Length of the rectangle
print("Maximum radius of the circle that can be packed and number:", max_circle_radius(w, l))

def get_values(x: [float], param: str):
    pass

def b(x: [float]):

    return get_values(x, "b-param")

def c(x: [float]):

    return get_values(x, "c-param")

def b_var(x: [float]):

    #return get_values(x, "b_var")
    return get_values(x, "b_var")


def c_var(x: [float]):

    return get_values(x, "c_var")


def b_constraint(x: [float]):

    return 1 - get_values(x, "b-param")


def c_constraint(x: [float]):

    return 15 - get_values(x, "c-param")


def b_var_constraint(x: [float]):

    return 10 - get_values(x, "b_var")


#bounds = {'sr': (0.001 * 5, 0.001 * 125), 'ht': (0.001 * 50, 0.001 * 100), 'cs': (0.001 * 25, 0.001 * 250), 'theta_deg': (0.0, 0.0)}# Bounds for optimization

problem = (
    OnTheFlyFloatProblem()
    .set_name("Testing")
    .add_variable(0.001 * 5, 0.001 * 125)
    .add_variable(0.001 * 50, 0.001 * 100)
    #.add_variable(0.001 * 25, 0.001 * 250)
    #.add_variable(0.001 * 25, 1.0)
    .add_variable(0.0, 0.0)
    .add_function(c)
    .add_function(b)
    .add_function(b_var)
    .add_function(c_var)
    .add_constraint(b_constraint)
    .add_constraint(c_constraint)
    .add_constraint(b_var_constraint)
)

if __name__ == "__main__":

    with open(file_home_path + "calc_log_obj.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "sr", "ht", "cs", "theta_deg", "b-param", "c-param", "b_var", "c_var","execution time", "step count"])
        file.close()

    max_evaluations = 640

    '''

    algorithm = NSGAII(
        population_evaluator=MultiprocessEvaluator(processes=16),
        problem=problem,
        population_size=16,
        offspring_population_size=16,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        #dominance_comparator=DominanceComparator(),
    )

    '''

    algorithm = GDE3(
        population_evaluator=MultiprocessEvaluator(processes=16),
        problem=problem,
        population_size=16,
        cr=0.9,
        f=0.8,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        dominance_comparator=DominanceComparator(),
    )


    algorithm.run()
    solutions = algorithm.get_result()
    front = algorithm.get_result()
    #front = get_non_dominated_solutions(solutions)

    for sol in range(len(front)):
        vars = front[sol].variables
        #print(f'(Solution #{sol + 1}): Variables={front[sol].variables}; Objectives={front[sol].objectives}')
        printing(f'(Solution #{sol + 1}): (Filename - {make_filename(float(vars[0]), float(vars[1]), float(vars[2]), float(vars[3]))})')
        printing(f'             Variables={vars}')
        printing(f'             Objectives={front[sol].objectives}')

    printing(f"Computing time: {algorithm.total_computing_time}")
