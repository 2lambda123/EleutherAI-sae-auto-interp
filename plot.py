# %%
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def process_directories(parent_dir, N_values):
    results = {}
    for N in N_values:
        directory = os.path.join(parent_dir, f"recall_{N}")
        times = []
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    times.append(data['time'])
        results[N] = times
    return results

def create_plot(results):
    N_values = list(results.keys())
    mean_times = [np.mean(times) if times else 0 for times in results.values()]
    std_times = [np.std(times) if times else 0 for times in results.values()]

    plt.figure(figsize=(10, 6))
    plt.errorbar(N_values, mean_times, yerr=std_times, fmt='-o', capsize=5, capthick=1, ecolor='lightgray', color='darkblue')
    
    plt.title('Time vs N')
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.xticks(N_values)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Fill the area between mean +/- std with a light color
    plt.fill_between(N_values, 
                     [m - s for m, s in zip(mean_times, std_times)],
                     [m + s for m, s in zip(mean_times, std_times)],
                     alpha=0.2, color='lightblue')

    plt.tight_layout()
    plt.show()

# Main execution
def main(parent_directory):
    N_values = [1, 5, 15, 35]
    results = process_directories(parent_directory, N_values)
    create_plot(results)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <parent_directory>")
        sys.exit(1)
    parent_directory = "results/scores"
    main(parent_directory)

# %%

import os
import json
import numpy as np
from scipy import stats

def process_directories(parent_dir, N_values):
    results = {}
    for N in N_values:
        # directory = os.path.join(parent_dir, f"recall_{N}")
        directory = os.path.join(parent_dir, f"simulation")
        times = []
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    times.append(data['time'])
        results[N] = times
    return results

def calculate_stats(times):
    if not times:
        return 0, 0, 0, (0, 0)
    
    mean = np.mean(times)
    std = np.std(times, ddof=1)  # ddof=1 for sample standard deviation
    sem = stats.sem(times)
    
    # Calculate 95% confidence interval
    df = len(times) - 1
    ci = stats.t.interval(0.95, df, loc=mean, scale=sem)
    
    return mean, std, sem, ci

def main(parent_directory):
    # N_values = [1, 5, 15, 35]
    N_values = [0]
    results = process_directories(parent_directory, N_values)
    
    print(f"{'N':<5}{'Mean':<10}{'Std Dev':<10}{'SEM':<10}{'95% CI':<20}")
    print("-" * 55)
    
    for N, times in results.items():
        mean, std, sem, ci = calculate_stats(times)
        print(f"{N:<5}|{mean:.4f}|{std:.4f}|{sem:.4f}|{ci[0]:.4f} - {ci[1]:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <parent_directory>")
        sys.exit(1)
    parent_directory = "results"
    main(parent_directory)


# %%

import re
import numpy as np

def process_log_file(file_path):
    completion_tokens = []
    prompt_tokens = []
    
    pattern = r'CompletionUsage\(completion_tokens=(\d+), prompt_tokens=(\d+), total_tokens=\d+\)'
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                completion_tokens.append(int(match.group(1)))
                prompt_tokens.append(int(match.group(2)))
    
    return completion_tokens, prompt_tokens

def calculate_stats(data):
    return np.mean(data), np.std(data)

def main(file_path):
    completion_tokens, prompt_tokens = process_log_file(file_path)
    
    if not completion_tokens or not prompt_tokens:
        print("No matching data found in the log file.")
        return
    
    completion_mean, completion_std = calculate_stats(completion_tokens)
    prompt_mean, prompt_std = calculate_stats(prompt_tokens)
    
    print(f"Completion Tokens:")
    print(f"  Average: {completion_mean:.2f}")
    print(f"  Standard Deviation: {completion_std:.2f}")
    print(f"\nPrompt Tokens:")
    print(f"  Average: {prompt_mean:.2f}")
    print(f"  Standard Deviation: {prompt_std:.2f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_log_file>")
        sys.exit(1)
    log_file_path = "token_stats/vllm.log"
    main(log_file_path)


# %%

import re
import numpy as np

def process_log_file(file_path):
    response_tokens = []
    prompt_tokens = []
    
    response_pattern = r'Response tokens: (\d+)'
    prompt_pattern = r'Prompt tokens: (\d+)'
    
    with open(file_path, 'r') as file:
        for line in file:
            response_match = re.search(response_pattern, line)
            prompt_match = re.search(prompt_pattern, line)
            
            if response_match:
                response_tokens.append(int(response_match.group(1)))
            elif prompt_match:
                prompt_tokens.append(int(prompt_match.group(1)))
    
    return response_tokens, prompt_tokens

def calculate_stats(data):
    return np.mean(data), np.std(data)

def main(file_path):
    response_tokens, prompt_tokens = process_log_file(file_path)
    
    if not response_tokens and not prompt_tokens:
        print("No matching data found in the log file.")
        return
    
    if response_tokens:
        response_mean, response_std = calculate_stats(response_tokens)
        print(f"Response Tokens:")
        print(f"  Average: {response_mean:.2f}")
        print(f"  Standard Deviation: {response_std:.2f}")
        print(f"  Count: {len(response_tokens)}")
    else:
        print("No Response Tokens data found.")
    
    if prompt_tokens:
        prompt_mean, prompt_std = calculate_stats(prompt_tokens)
        print(f"\nPrompt Tokens:")
        print(f"  Average: {prompt_mean:.2f}")
        print(f"  Standard Deviation: {prompt_std:.2f}")
        print(f"  Count: {len(prompt_tokens)}")
    else:
        print("\nNo Prompt Tokens data found.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_log_file>")
        sys.exit(1)
    log_file_path = "vllm.log"
    main(log_file_path)

