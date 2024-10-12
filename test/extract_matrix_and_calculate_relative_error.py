import re
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Constants
sigma_sm_di_higgs = 1.454e+01
BR_hh_4b = 0.53 * 0.53
L = 3000
sigma_b = np.array([2.237e+05, 9.370e+01, 5.996e+03, 2.506e+01, 2.475e+06, 5.67e+08, 2.135e+02, 4.647e+01])

def calculate_with_confusion_matrix(confusion_matrix):
    # Extract ps and pb
    ps = confusion_matrix[0, 0]
    pb = confusion_matrix[1:, 0]

    # Calculate the numerator
    numerator = ps * sigma_sm_di_higgs * BR_hh_4b * L + np.sum(pb * sigma_b * L)

    # Calculate the denominator
    denominator = ps * BR_hh_4b * L * sigma_sm_di_higgs

    # Calculate the final result and multiply by 100%
    result = (np.sqrt(3.841 * numerator) / denominator) * 100

    return result

def extract_confusion_matrices(log_file_path):
    with open(log_file_path, 'r') as f:
        log_data = f.readlines()

    confusion_matrices = []
    inside_matrix = False
    current_matrix = []
    current_row = []

    # Iterate through each line of the log file
    for line in log_data:
        # Check if this is the start of a confusion matrix
        if "- confusion_matrix:" in line:
            inside_matrix = True
            current_matrix = []
            current_row = []
            continue
        
        # If we are inside a confusion matrix, parse numbers
        if inside_matrix:
            # Extract numbers from the line (both scientific notation and standard floats)
            numbers = re.findall(r"[-+]?\d+\.\d+e[-+]?\d+|\d+\.\d+", line)
            if numbers:
                current_row.extend([float(num) for num in numbers])
            
            # If 9 numbers are collected, treat it as a row of the matrix
            if len(current_row) >= 9:
                current_matrix.append(current_row[:9])  # Take the first 9 numbers
                current_row = current_row[9:]  # Retain any remaining numbers for the next row
                
            # Once 9 rows are collected, the matrix is complete
            if len(current_matrix) == 9:
                confusion_matrices.append(np.array(current_matrix))
                inside_matrix = False  # Done reading the matrix
                current_matrix = []  # Prepare for the next matrix

    return confusion_matrices

def plot_results(results):
    epochs = list(range(1, len(results) + 1))  # Create epoch sequence
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, results, marker='o', linestyle='-', color='b', label='Relative Error (%)')
    plt.xlabel('Epoch')  # Set x-axis label to epoch
    plt.ylabel('Relative Error (%)')  # Set y-axis label to relative error
    plt.title('Relative Error over Epochs')
    plt.xticks(epochs)  # Set x-axis ticks to show all epochs
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Extract confusion matrices from a log file and perform calculations.")
    
    # Add argument for the log file path
    parser.add_argument('log_file_path', type=str, help='Path to the log file containing the confusion matrices')

    # Parse command line arguments
    args = parser.parse_args()
    
    # Get log file path from arguments
    log_file_path = args.log_file_path

    # Extract confusion matrices from the log file
    confusion_matrices = extract_confusion_matrices(log_file_path)

    # List to store all the results
    results = []

    # Perform calculation for each confusion matrix and store the result
    for i, matrix in enumerate(confusion_matrices):
        result = calculate_with_confusion_matrix(matrix)
        results.append(result)
        print(f"Result for Confusion Matrix {i+1}: {result}%")

    # Plot the results
    plot_results(results)

