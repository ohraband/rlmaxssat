import random
import math
import os
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def read_input(filename):
    weights = []
    clauses = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('w'):
            weights = list(map(int, line.strip().split()[1:-1]))
        elif not line.startswith('c') and not line.startswith('p'):
            clauses.append(list(map(int, line.strip().split()[:-1])))
    num_variables = len(weights)
    return num_variables, clauses

def evaluate_solution(solution, clauses):
    total_satisfied = 0
    for clause in clauses:
        if clause_satisfied(solution, clause):
            total_satisfied += 1
    return total_satisfied

def clause_satisfied(solution, clause):
    for literal in clause:
        var_index = abs(literal) - 1
        if (literal > 0 and solution[var_index] == 1) or (literal < 0 and solution[var_index] == 0):
            return True
    return False

def generate_neighbor(solution):
    neighbor = solution[:]
    index = random.randint(0, len(solution) - 1)
    neighbor[index] = 1 - neighbor[index]
    return neighbor

def simulated_annealing_maxsat(
        clauses, num_variables, initial_temperature,
        max_neighbors=40, cooling_rate=0.99):

    current_solution = [random.randint(0, 1) for _ in range(num_variables)]
    current_clauses = evaluate_solution(current_solution, clauses)

    temperature_iterations = 0
    states_generated = 1
    total_flips = 0

    best_solution = current_solution[:]
    best_clauses = current_clauses

    current_temperature = initial_temperature
    max_clauses = len(clauses)

    while current_temperature > 0.1:

        temperature_iterations += 1

        for _ in range(max_neighbors):

            neighbor_solution = generate_neighbor(current_solution)
            neighbor_clauses = evaluate_solution(neighbor_solution, clauses)
            states_generated += 1
            delta = neighbor_clauses - current_clauses

            if delta > 0 or random.random() < math.exp(delta / current_temperature):
                current_solution = neighbor_solution
                current_clauses = neighbor_clauses
                total_flips += 1

                if current_clauses > best_clauses:
                    best_solution = current_solution[:]
                    best_clauses = current_clauses

                # Early stopping if perfect solution found
                if best_clauses == max_clauses:
                    return best_solution, best_clauses, temperature_iterations, states_generated, total_flips

        current_temperature *= cooling_rate

    return best_solution, best_clauses, temperature_iterations, states_generated, total_flips


def calculate_initial_temperature(clauses, num_variables, num_samples=1000, cooling_rate=0.99):
    total_change = 0
    for _ in range(num_samples):
        sol1 = [random.randint(0, 1) for _ in range(num_variables)]
        sol2 = generate_neighbor(sol1)
        delta = evaluate_solution(sol2, clauses) - evaluate_solution(sol1, clauses)
        total_change += abs(delta)
    average_change = total_change / num_samples
    return -average_change / math.log(cooling_rate)


def plot_cumulative(data, title, xlabel, ylabel, output_file):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(data)+1), data, linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def plot_satisfied_clauses_histogram(satisfied_list, output_file, max_clauses=None, margin=3):
    if not max_clauses:
        max_clauses = max(satisfied_list)
    min_range = max(0, max_clauses - margin)
    max_range = max_clauses + margin

    freq = {}
    for val in satisfied_list:
        if min_range <= val <= max_range:
            freq[val] = freq.get(val, 0) + 1

    x_values = list(range(min_range, max_range + 1))
    counts = [freq.get(x, 0) for x in x_values]

    plt.figure(figsize=(10, 5))
    plt.bar(x_values, counts, color='skyblue', edgecolor='black')
    plt.title("Number of Problems per Satisfied Clauses Count")
    plt.xlabel("Satisfied Clauses")
    plt.ylabel("Number of Problems")
    plt.grid(axis='y')
    plt.xticks(x_values)
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    folder_path = "wuf20-71R-R"
    problemset_name = os.path.basename(folder_path)
    mwcnf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mwcnf")]

    output_folder = "plots"
    os.makedirs(output_folder, exist_ok=True)

    max_neighbors = 40
    cooling_rate = 0.995

    total_problems = 0
    total_fully_satisfied = 0

    satisfied_clauses_list = []
    temp_iterations_list = []
    accepted_flips_list = []

    for file_path in mwcnf_files:
        num_variables, clauses = read_input(file_path)
        initial_temperature = calculate_initial_temperature(clauses, num_variables, 1000, cooling_rate)

        best_solution, best_clauses, temp_iters, states_generated, flips = simulated_annealing_maxsat(
            clauses, num_variables, initial_temperature,
            max_neighbors, cooling_rate
        )

        satisfied_clauses_list.append(best_clauses)
        temp_iterations_list.append(temp_iters)
        accepted_flips_list.append(flips)

        total_problems += 1
        if best_clauses == len(clauses):
            total_fully_satisfied += 1

        print(f"Problem: {file_path}")
        print(f"  Satisfied Clauses: {best_clauses}/{len(clauses)}")
        print(f"  Temperature Iterations: {temp_iters}")
        print(f"  Total Accepted Flips: {flips}")

    # --- File names based on problemset ---
    temp_plot_file = os.path.join(output_folder, f"{problemset_name}_temp_iterations.png")
    flips_plot_file = os.path.join(output_folder, f"{problemset_name}_accepted_flips.png")
    clauses_plot_file = os.path.join(output_folder, f"{problemset_name}_satisfied_clauses_histogram.png")
    avg_metrics_file = os.path.join(output_folder, f"{problemset_name}_average_metrics.txt")

    # --- Plots ---
    plot_cumulative(
        temp_iterations_list,
        "Temperature Iterations per Problem",
        "Problem Index",
        "Temperature Iterations",
        temp_plot_file
    )

    plot_cumulative(
        accepted_flips_list,
        "Accepted Flips per Problem",
        "Problem Index",
        "Accepted Flips",
        flips_plot_file
    )

    plot_satisfied_clauses_histogram(
        satisfied_clauses_list,
        clauses_plot_file,
        max_clauses=max(satisfied_clauses_list),
        margin=3
    )

    # --- Summary ---
    satisfied_ratio = total_fully_satisfied / total_problems
    avg_temp_iterations = sum(temp_iterations_list) / total_problems
    avg_accepted_flips = sum(accepted_flips_list) / total_problems
    avg_satisfied_clauses = sum(satisfied_clauses_list) / total_problems

    print(f"\nRatio of fully satisfied problems: {satisfied_ratio:.2f}")
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Total Problems", total_problems])
    table.add_row(["Fully Satisfied Ratio", f"{satisfied_ratio:.2f}"])
    table.add_row(["Average Temperature Iterations", f"{avg_temp_iterations:.2f}"])
    table.add_row(["Average Accepted Flips", f"{avg_accepted_flips:.2f}"])
    table.add_row(["Average Clauses Satisfied", f"{avg_satisfied_clauses:.2f}"])
    print(table)

    # --- Save averages to txt ---
    with open(avg_metrics_file, "w") as f:
        f.write(f"Problem Set: {problemset_name}\n")
        f.write(f"Total Problems: {total_problems}\n")
        f.write(f"Fully Satisfied Ratio: {satisfied_ratio:.4f}\n")
        f.write(f"Average Temperature Iterations: {avg_temp_iterations:.2f}\n")
        f.write(f"Average Accepted Flips: {avg_accepted_flips:.2f}\n")
        f.write(f"Average Clauses Satisfied: {avg_satisfied_clauses:.2f}\n")


