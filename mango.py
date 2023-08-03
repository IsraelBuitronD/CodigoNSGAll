import functools
import math
import matplotlib.pyplot as plt
import numpy as np

#----------Global Variables-----------

population_size = 50  # Size of the population
num_generations = 600  # Number of generations to run
mutation_rate = 0.05  # Mutation rate for evolution
crossover_rate = 1.0  # Crossover rate for evolution

min_value = -5  # Minimum value of the search range
max_value = 5  # Maximum value of the search range
n = 3  # Number of variables of the objective function
m = 3  # Number of objective functions

#---------------------------------------

def plot_2d_points(pt):
    """
    Plot points in 2D.

    Args:
        pt (numpy.ndarray): Array containing the points to be plotted.

    """

    # The [:n_poblacion, 0] notation selects the first column of points (x-coordinate)
    # and [:n_poblacion, 1] selects the second column (y-coordinate).
    plt.plot(pt[:population_size, 0], pt[:population_size, 1], ".")

    # The function pause takes a parameter in seconds, so 0.001 is equivalent to one millisecond.
    plt.pause(0.001)

    # Draw the plot.
    plt.draw()

    # Clear the current axis (plot) to prepare for the next plot.
    plt.cla()

def plot_3d_points(pt, ax):
    """
    Plot points in 3D.

    Args:
        pt (numpy.ndarray): Array containing the points to be plotted.
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): Matplotlib's 3D Axes object.

    Returns:
        None
    """

    # The [:n_poblacion, 0] notation selects the first column of points for the 'x' axis
    # [:n_poblacion, 1] selects the second column for the 'y' axis, and [:n_poblacion, 2]
    # selects the third column for the 'z' axis.
    xg = pt[:population_size, 0]
    yg = pt[:population_size, 1]
    zg = pt[:population_size, 2]

    # Plot the points in 3D using the provided Axes3D object.
    ax.plot(xg, yg, zg, ".")

    # The function pause takes a parameter in seconds, so 0.001 is equivalent to one millisecond.
    plt.pause(0.001)

    # Draw the plot.
    plt.draw()

    # Clear the current axis (plot) to prepare for the next plot.
    plt.cla()

def pareto_dominance(a, b):
    """
    Checks if point 'a' dominates point 'b' in terms of Pareto dominance.

    Args:
        a (numpy.ndarray): Array with the coordinates of point 'a'.
        b (numpy.ndarray): Array with the coordinates of point 'b'.

    Returns:
        bool: True if point 'a' dominates 'b', False otherwise.
    """

    # Define the first condition according to Pareto dominance.
    # Each element of 'a' must be less than or equal to each element of 'b'.
    condition_1 = a <= b

    # Define the second condition.
    # At least one element of 'a' must be strictly less than the corresponding elements of 'b'.
    condition_2 = a < b

    # Check both conditions, 'a' dominates 'b' if condition 1 holds for all elements and condition 2 for at least one.
    evaluation = np.all(condition_1) and np.any(condition_2)

    # If both conditions are satisfied, return True; otherwise, return False.
    if evaluation:
        return True
    return False


def pareto_dominance_matrix(data):
    """
    Calculates the Pareto dominance matrix for a set of data points.

    Args:
        data (numpy.ndarray): Array containing the data points to be analyzed. Each row represents a point.

    Returns:
        numpy.ndarray: Boolean matrix indicating if each point dominates the other points or not.
    """
    # Get the number of points (p_t) and the number of dimensions (n) of the data
    p_t = data.shape[0]
    n = data.shape[1]

    # Create two arrays of zeros, each with shape (p_t, p_t, n).
    # These arrays will be used to compare each point with all other points in the different dimensions.
    x = np.zeros([p_t, p_t, n])
    y = np.zeros([p_t, p_t, n])

    # Fill the arrays x and y with the input data to be able to compare them in the following stages.
    for i in range(p_t):
        x[i, :, :] = data[i]
        y[:, i, :] = data[i]

    # Create two boolean conditions: condition_1 and condition_2.
    # condition_1 evaluates if each element of x is less than or equal to the corresponding element of y.
    # condition_2 evaluates if any element of x is strictly less than the corresponding element of y.
    condition_1 = x <= y
    condition_2 = x < y

    # Evaluate if all elements are better or equal.
    all_better_or_equal = np.all(condition_1, axis=2)

    # Evaluate if any element is strictly better
    any_strictly_better = np.any(condition_2, axis=2)

    # Apply a logical AND operation between np.all(condition_1, axis=2) and np.any(condition_2, axis=2)
    pareto_dominance_matrix = np.logical_and(all_better_or_equal, any_strictly_better)

    # The resulting matrix is a boolean matrix where the element in row i and column j is True
    # if point i dominates point j, and False otherwise.
    return pareto_dominance_matrix



def pareto_fronts(data):
    """
    Calculates the Pareto fronts for a set of data.

    Args:
        data (numpy.ndarray): Array containing the data points to be analyzed. Each row represents a point.

    Returns:
        List[numpy.ndarray]: List of arrays with the indices of the points in each Pareto front.
    """

    # Create an empty list to store the dominance sets of each point
    dominance_sets = []
    # Create an empty list to count how many points each point dominates
    point_counts = []

    # Calculate the dominance matrix for the input data
    dominance_matrix = pareto_dominance_matrix(data)
    # Get the number of points
    pop_size = data.shape[0]

    # Iterate through all points and calculate their dominance set and the number of points they dominate
    for i in range(pop_size):
        current_dominance_set = set()
        point_counts.append(0)
        for j in range(pop_size):
            if dominance_matrix[i, j]:
                current_dominance_set.add(j)
            elif dominance_matrix[j, i]:
                point_counts[-1] += 1
        dominance_sets.append(current_dominance_set)

    # Convert the list of counts to a numpy array to perform mathematical operations with it
    point_counts = np.array(point_counts)
    # Create an empty list to store the Pareto fronts
    pareto_fronts = []
    # Iterate until there are no more points in the Pareto fronts
    while True:
        # Find the points that are not dominated by any other
        current_front = np.where(point_counts == 0)[0]
        # If there are no points that are not dominated, exit the loop
        if len(current_front) == 0:
            break
        # Add the points of the current front to the list of Pareto fronts
        pareto_fronts.append(current_front)

        # Mark the points of the current front as dominated (-1 in the point counts list)
        for individual in current_front:
            point_counts[individual] = -1
            # Iterate through the points that the current point dominated and reduce their count by 1
            current_dominated_set = dominance_sets[individual]
            for dominated_point in current_dominated_set:
                point_counts[dominated_point] -= 1
    return pareto_fronts


def crowding(data, fronts):
    """
    Calculates the crowding distance measure for each individual in a population.

    Args:
        data (np.array): Data matrix, where each row represents an individual, and each column represents a variable.
        fronts (list): List of Pareto fronts, where each front is a list of indices of individuals in the data.

    Returns:
        crowding_measure (np.array): Array of crowding distance measures for each individual in the data.
    """

    columns = data.shape[1]  # number of columns, i.e., the number of objective functions
    rows = data.shape[0]  # number of rows, i.e., the number of individuals in the population

    # Calculate the crowding distance of each individual for each objective function
    crowding_distance = np.zeros_like(data)
    for n in range(columns):
        f_min = np.min(data[:, n])  # minimum value of the n-th column (objective function)
        f_max = np.max(data[:, n])  # maximum value of the n-th column (objective function)
        v = f_max - f_min  # range of the n-th objective function
        crowding_distance[:, n] = (data[:, n] - f_min) / v  # crowding distance of each individual for the n-th objective function

    data = crowding_distance  # reassign crowding distances to the 'data' variable
    crowding_measure = np.zeros(rows)  # initialization of the crowding distance measure for each individual

    # Calculate the crowding distance measure for each individual in each Pareto front
    for front in fronts:
        for n in range(columns):
            sorted_front = sorted(front, key=lambda x: data[x, n])  # sort the Pareto front by crowding distance for the n-th objective function
            crowding_measure[sorted_front[0]] = np.inf  # set the crowding distance measure of the first individual in the Pareto front to infinity (boundary solution)
            crowding_measure[sorted_front[-1]] = np.inf  # set the crowding distance measure of the last individual in the Pareto front to infinity (boundary solution)
            if len(sorted_front) > 2:
                # calculate the crowding distance measure for individuals that are not boundary solutions
                for i in range(1, len(sorted_front) - 1):
                    crowding_measure[sorted_front[i]] += data[sorted_front[i + 1], n] - data[sorted_front[i - 1], n]

    return crowding_measure

def rank(fronts):
    """
    Returns a dictionary containing the rank index for each individual in the given Pareto fronts.

    Args:
        fronts (list): A list of lists of indices of individuals belonging to each Pareto front.

    Returns:
        rank_index (dict): A dictionary containing the rank index for each individual.
        The rank index is an integer that indicates the level of the Pareto front to which the individual belongs
        (individuals in the first front have a rank index of 0, those in the second front have a rank index of 1, and so on).
    """

    rank_index = {}  # Dictionary to store the rank index of each individual
    for i, front in enumerate(fronts):  # Iterate through each Pareto front and assign the rank index to each individual
        for x in front:
            rank_index[x] = i

    return rank_index

def non_dominated_sort(rank_index, crowding):
    """
    Function that performs non-dominated sorting of a set of solutions using the crowding distance method.

    Args:
        rank_index (list): List with the ranking indices of the solutions.
        crowding (list): List with the crowding distance values of the solutions.

    Returns:
        list: List with the indices of the solutions sorted by non-dominated order, from best to "worst".
    """
    num_rows = len(crowding)
    indices = list(range(num_rows))

    def non_dominated_rank(a, b):
        if rank_index[a] > rank_index[b]:
            return -1  # -1 if bdom a and b is less crowded
        elif rank_index[a] < rank_index[b]:
            return 1  # if a dominates b and a is less crowded
        else:
            if crowding[a] < crowding[b]:
                return -1
            elif crowding[a] > crowding[b]:
                return 1
            else:
                return 0  # if they are equal in all respects


    #reverse indicates that a descending sorting will be performed.
    # Sorted indices based on non-dominated order, from best to "worst"
    sorted_indices = sorted(indices, key=functools.cmp_to_key(non_dominated_rank), reverse=True)

    return sorted_indices


def selection(population_size, q_t):
    """
    Selects q_t individuals randomly from a population of size 'population_size'.

    Args:
    - population_size (int): Size of the population.
    - q_t (int): Number of individuals to select.

    Returns:
    - selected (List[int]): List of indices of the selected individuals.
    """
    selected = []  # Initializes the list of indices of selected individuals
    for _ in range(q_t):  # Repeat the process q_t times
        index = np.random.randint(0, population_size)  # Choose a random index between 0 and population_size (exclusive)
        parent = index  # Choose the index as the parent
        selected.append(parent)  # Add the parent index to the list of selected indices

    return selected  # Return the list of selected indices


def crossover(individuals, rate):
    """
    Performs the crossover of individuals according to a crossover rate.

    Args:
        individuals (numpy.ndarray): Two-dimensional array with the individuals to be crossed.
        rate (float): Crossover rate between 0 and 1.

    Returns:
        numpy.ndarray: Two-dimensional array with the offspring generated by the crossover.
    """
    # Copy the individuals to avoid modifying the original array
    offspring = individuals.copy()

    # Perform crossover for half of the rate
    for i in range(int(rate/2)):

        # Choose a first individual randomly
        x = np.random.randint(0, individuals.shape[0])
        # Choose a second individual randomly
        y = np.random.randint(0, individuals.shape[0])
        # Verify that the indices are not equal to avoid crossing an individual with itself
        while x == y:
            x = np.random.randint(0, individuals.shape[0])
            y = np.random.randint(0, individuals.shape[0])

        # Choose a crossover point randomly
        crossover_point = np.random.randint(1, individuals.shape[1])

        # Create offspring 1 with the first part of the first parent and the second part of the second parent
        offspring[2*i, 0:crossover_point] = individuals[x, 0:crossover_point]
        offspring[2*i, crossover_point:] = individuals[y, crossover_point:]
        # Create offspring 2 with the first part of the second parent and the second part of the first parent
        offspring[2*i+1, 0:crossover_point] = individuals[y, 0:crossover_point]
        offspring[2*i+1, crossover_point:] = individuals[x, crossover_point:]
    return offspring

def mutation(individual, min_val, max_val, mutation_rate):
    """
    Performs the mutation of an individual by adding Gaussian noise.

    Args:
        individual (numpy.ndarray): The individual to mutate.
        min_val (float): The minimum possible value for the elements of the individual.
        max_val (float): The maximum possible value for the elements of the individual.
        mutation_rate (float): The mutation rate, a number between 0 and 1 that indicates the probability of mutation.

    Returns:
        offspring (numpy.ndarray): The mutated individual.
    """
    mutation_prob = (max_val - min_val) * mutation_rate  # Calculate the probability of mutation
    offspring = individual.copy()  # Copy the individual
    # Generate Gaussian noise for the mutation
    offspring += np.random.normal(0, mutation_prob, size=offspring.shape)
    # Limit the values of the individual to the allowed minimum and maximum values
    offspring = np.clip(offspring, min_val, max_val)
    return offspring


def NSGA_II(current_population, objectives, generation):
    """
    Implementation of the NSGA-II algorithm for multi-objective optimization.

    Args:
        current_population (np.array): Current population.
        objectives (np.array): Objective values corresponding to the current population.
        generation (int): Current generation number.

    Returns:
        np.array: Population of the next generation.
    """

    # Print information about the current generation
    print(f"NSGA-II in generation {generation + 1}")

    # Get the Pareto fronts of the current population
    fronts = pareto_fronts(objectives)

    # Get the position of each individual in its front
    rank_index = rank(fronts)

    # Calculate the crowding distance of each individual
    crowding_distance = crowding(objectives, fronts)

    # Sort the individuals by ranking and crowding distance
    non_dominated_indices = non_dominated_sort(rank_index, crowding_distance)

    # Select the best survivors for the next generation
    survivors = current_population[non_dominated_indices[:population_size]]

    # Select parents for the crossover
    selected_parents = selection(population_size, q_t=population_size)

    # Perform crossover with the selected parents
    crossover_t = crossover(selected_parents, rate=crossover_rate)

    # Perform mutation based on the obtained survivors
    next_population = np.array([mutation(survivors[i], min_value, max_value, mutation_rate) for i in crossover_t])

    # Combine the survivors and the mutated offspring to form the next generation population
    next_generation_population = np.concatenate([survivors, next_population])

    # Return the population of the next generation
    return next_generation_population


def sum_fonseca(x, n, f):
    """
    Calculates the sum of the Fonseca and Fleming function.

    Args:
        x (np.array): Input values.
        n (int): Dimension of the problem.
        f (str): Name of the function (f1 or f2).

    Returns:
        float: Sum of the Fonseca and Fleming function.
    """
    s = 0
    for i in range(n):
        if f == "f1":
            s += pow((x[:, i] - (1 - np.sqrt(n))), 2)
        elif f == "f2":
            s += pow((x[:, i] + (1 - np.sqrt(n))), 2)
    return s

def fonseca(x, n):
    """
    Fonseca and Fleming function for multi-objective optimization.

    Args:
        x (np.array): Input values.
        n (int): Dimension of the problem.

    Returns:
        np.array: Objective values corresponding to the input values.
    """
    f1 = 1 - np.exp(-sum_fonseca(x, n, "f1"))
    f2 = 1 - np.exp(-sum_fonseca(x, n, "f2"))

    return np.stack([f1, f2], axis=1)

def sine_function(f):
    """
    Function that applies the sine function to each value in a list.

    Args:
        f (list): List of values to which the sine function will be applied.

    Returns:
        list: List of values with the sine function applied to each of them.
    """
    for i in range(len(f)):
        f[i] = math.sin(10 * math.pi * f[i])
    return f


def g_function(x, n):
    """
    Auxiliary function for the ZDT3 function.

    Args:
        x (np.array): Input values.
        n (int): Dimension of the problem.

    Returns:
        float: Value of the auxiliary function.
    """
    g = 0
    for i in range(n):
        g += 1 + (9 / 29) * (x[:, i])
    return g

def h1_function(f, g):
    """
    Calculates the value of h1 based on f and g.

    Args:
        f: Value of the variable f.
        g: Value of the auxiliary variable g.

    Returns:
        Value of the auxiliary variable h1.
    """
    h = 1 - np.sqrt(f/g)
    return h

def h2_function(f, g):
    """
    Calculates the value of h2 based on f and g.

    Args:
        f: Value of the variable f.
        g: Value of the auxiliary variable g.

    Returns:
        Value of the auxiliary variable h2.
    """
    h = 1 - pow((f/g), 2)
    return h

def h3_function(f, g):
    """
    Calculates the value of h3 based on f and g.

    Args:
        f: Value of the variable f.
        g: Value of the auxiliary variable g.

    Returns:
        Value of the auxiliary variable h3.
    """
    h = 1 - np.sqrt(f/g) - ((f/g) * np.sin(f))
    return h

def f_variable(x):
    """
    Gets the value of the variable f from the input matrix x.

    Args:
        x: Matrix of values.

    Returns:
        Value of the variable f.
    """
    f = x[:, 1]
    return f

def zdt1_function(x, n):
    """
    Calculates the Zitzler-Deb-Thiele function #1 based on x and n.

    Args:
        x: Input matrix of values.
        n: Number of input variables.

    Returns:
        Matrix with the output values of the function.
    """
    f1 = f_variable(x)
    f2 = g_function(x, n) * h1_function(f_variable(x), g_function(x, n))
    return np.stack([f1, f2], axis=1)

def zdt2_function(x, n):
    """
    Calculates the Zitzler-Deb-Thiele function #2 based on x and n.

    Args:
        x: Input matrix of values.
        n: Number of input variables.

    Returns:
        Matrix with the output values of the function.
    """
    f1 = f_variable(x)
    f2 = g_function(x, n) * h2_function(f_variable(x), g_function(x, n))
    return np.stack([f1, f2], axis=1)

def zdt3_function(x, n):
    """
    Calculates the Zitzler-Deb-Thiele function #3 based on x and n.

    Args:
        x: Input matrix of values.
        n: Number of input variables.

    Returns:
        Matrix with the output values of the function.
    """
    f1 = f_variable(x)
    f2 = g_function(x, n) * h3_function(f_variable(x), g_function(x, n))
    return np.stack([f1, f2], axis=1)

def omega_function(gx, xi):
    """
    Calculates the omega function based on gx and xi.

    Args:
        gx: Value of gx.
        xi: Value of xi.

    Returns:
        Value of the omega function.
    """
    return np.pi / (4 * (1 + gx)) * (1 + 2 * gx * xi)


def dtlz1(x):
    """
    Calculate the DTLZ1 function based on the input matrix x.

    Args:
    x: input matrix of values

    Returns:
    Matrix with the output values of the function
    """
    print("DTLZ1")

    functions = []

    m = 3  # number of functions

    gx = 100 * (np.sum(np.square(x[:, m:] - 0.5) - np.cos(20 * np.pi * (x[:, m:] - 0.5)), axis=1))

    for f in range(m):
        if f == m - 1:
            xi = (1 - x[:, 0])  # (1 - x1)
        else:
            xi = x[:, 0]  # x1

            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * x[:, (i + 1)]

            if f == 0:
                xi = xi
            else:
                xi = xi * (1 - x[:, ((m - 1) - f)])  # (1 - Xm-1)

        fi = 0.5 * (xi * (1 + gx))  # fm(x)

        functions.append(fi)

    return np.stack(functions, axis=1)

def dtlz2(x):
    """
    Calculate the DTLZ2 function based on the input matrix x.

    Args:
    x: input matrix of values

    Returns:
    Matrix with the output values of the function
    """
    print("DTLZ2")

    functions = []

    m = 3  # number of functions

    gx = np.sum(np.square(x[:, m:] - 0.5), axis=1)

    for f in range(m):
        if f == m - 1:
            xi = np.sin(x[:, 0] * (np.pi) / 2)  # sin(x1) pi/2
        else:
            xi = np.cos(x[:, 0] * (np.pi) / 2)  # x1

            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * (np.cos(x[:, i + 1] * (np.pi) / 2))

            if f == 0:
                xi = xi * (np.cos(x[:, (m - 2) - f] * (np.pi) / 2))
            else:
                xi = xi * (np.sin(x[:, (m - 1) - f] * (np.pi) / 2))  # (1 - Xm-1)

        fi = (1 + gx) * xi  # fm(x)

        functions.append(fi)

    return np.stack(functions, axis=1)

def dtlz3(x):
    """
    Calculate the DTLZ3 function based on the input matrix x.

    Args:
    x: input matrix of values

    Returns:
    Matrix with the output values of the function
    """
    print("DTLZ3")

    functions = []

    m = 3  # number of functions

    gx = 100 * (np.sum(np.square(x[:, m:] - 0.5) - np.cos(20 * np.pi * (x[:, m:] - 0.5)), axis=1))

    for f in range(m):
        if f == m - 1:
            xi = (np.sin(x[:, 0] * (np.pi) / 2))  # (sin(x1) pi/2)
        else:
            xi = np.cos(x[:, 0] * (np.pi) / 2)  # x1

            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi  = xi * (np.cos(x[:, i + 1] * (np.pi) / 2))

            if f == 0:
                xi = xi * (np.cos(x[:, (m - 2) - f] * (np.pi) / 2))
            else:
                xi = xi * (np.sin(x[:, (m - 1) - f] * (np.pi) / 2))  # (1 - Xm-1)

        fi = (1 + gx) * xi  # fm(x)

        functions.append(fi)

    return np.stack(functions, axis=1)

def dtlz4(x):
    """
    Calculate the DTLZ4 function based on the input matrix x.

    Args:
    x: input matrix of values

    Returns:
    Matrix with the output values of the function
    """
    print("DTLZ4")

    m = 3  # number of functions

    gx = np.sum(np.square(x[:, m:] - 0.5), axis=1)

    functions = []

    for f in range(m):
        if f == m - 1:
            xi = np.sin(np.power(x[:, 0], 10) * np.pi / 2)  # (sin(x1) pi/2)
        else:
            xi = np.cos(np.power(x[:, 0], 10) * np.pi / 2)  # x1
            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * np.cos(np.power(x[:, i + 1], 10) * np.pi / 2)
            if f == 0:
                xi = xi * np.cos(np.power(x[:, (m - 2) - f], 10) * np.pi / 2)
            else:
                xi = xi * np.sin(np.power(x[:, (m - 1) - f], 10) * np.pi / 2)  # (1 - Xm-1)
        fi = (1 + gx) * xi  # fm(x)
        functions.append(fi)

    return np.stack(functions, axis=1)


def dtlz5(x):
    """
    Calculate the DTLZ5 function based on the input matrix x.

    Args:
    x: input matrix of values

    Returns:
    Matrix with the output values of the function
    """
    print("DTLZ5")

    functions = []

    gx = np.sum(np.square(x[:, m:] - 0.5), axis=1)

    for f in range(m):  # m number of functions
        if f == m - 1:
            xi = (np.sin(omega_function(gx, x[:, 0]) * (np.pi) / 2))  # ( sin(x1) pi/2 )
        else:
            xi = np.cos(omega_function(gx, x[:, 0]) * (np.pi) / 2)  # x1

            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * (np.cos(omega_function(gx, x[:, i + 1]) * (np.pi) / 2))

            if f == 0:
                xi = xi * (np.cos(omega_function(gx, x[:, ((m - 2) - f)]) * (np.pi) / 2))
            else:
                xi = xi * (np.sin(omega_function(gx, x[:, ((m - 1) - f)]) * (np.pi) / 2))  # (1 - Xm-1)

        fi = (1 + gx) * xi  # fm(x)

        functions.append(fi)

    return np.stack(functions, axis=1)

def dtlz6(x):
    """
    Calculate the DTLZ6 function based on the input matrix x.

    Args:
    x: input matrix of values

    Returns:
    Matrix with the output values of the function
    """
    print("DTLZ6")

    functions = []

    gx = np.sum(np.power(x[:, m:], 0.1), axis=1)

    for f in range(m):  # m number of functions
        if f == m - 1:
            xi = np.sin(omega_function(gx, x[:, 0]) * (np.pi) / 2)  # sin(x1 * pi/2)
        else:
            xi = np.cos(omega_function(gx, x[:, 0]) * (np.pi) / 2)  # x1

            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi *= np.cos(omega_function(gx, x[:, i + 1]) * (np.pi) / 2)

            if f == 0:
                xi *= np.cos(omega_function(gx, x[:, (m - 2) - f]) * (np.pi) / 2)
            else:
                xi *= np.sin(omega_function(gx, x[:, (m - 1) - f]) * (np.pi) / 2)  # (1 - Xm-1)

        fi = (1 + gx) * xi  # fm(x)

        functions.append(fi)

    return np.stack(functions, axis=1)

def kursawe(x):
    """Calculate the objective values of the Kursawe function.

    Args:
        x (array): input matrix with the decision variables.

    Returns:
        array: matrix with the calculated objective values.
    """
    sq_x = x ** 2
    objective_1 = -10 * np.exp(-0.2 * np.sqrt(np.sum(sq_x[:, :2], axis=1))) - 10 * np.exp(-0.2 * np.sqrt(np.sum(sq_x[:, 1:], axis=1)))
    objective_2 = np.sum(np.power(np.abs(x), 0.8) + 5 * np.sin(x ** 3), axis=1)

    return np.stack([objective_1, objective_2], axis=1)

def main():
    """
    Main function that starts the program.
    """
    print("Hello, starting...")

    # Create the initial population p0
    x = np.random.uniform(min_value, max_value, (2 * population_size, n))
    plt.ion()

    # ax = plt.axes(projection="3d")
    # ax.view_init(45, 45)

    for generation in range(num_generations):
        pt = kursawe(x)
        x = NSGA_II(x, pt, generation)
        # plot(pt, ax)
        plot_2d_points(pt)

    plt.ioff()

    plt.plot(pt[:population_size, 0], pt[:population_size, 1], ".", color="r")
    plt.show()

    # Test comment

    """
    ax = plt.axes(projection="3d")
    ax.view_init(45, 45)
    xg = pt[:n_poblacion, 0]
    yg = pt[:n_poblacion, 1]
    zg = pt[:n_poblacion, 2]

    ax.plot(xg, yg, zg, ".")
    plt.show()
    """

main()
