import matplotlib.pyplot as plt
import json
import os
import numpy as np


def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def extract_info_from_filename(file_name):
    parts = file_name.split('_')
    if len(parts) >= 4:
        return " ".join(parts[1:4]).replace('.json', '')
    else:
        return "Unknown"


def get_common_files(dir1, dir2, dir3):
    # List directories in dir1 and dir2
    dir1_folders = set(next(os.walk(dir1))[1])
    dir2_folders = set(next(os.walk(dir2))[1])

    # List json files in dir3
    json_files = {file[:-5].replace("_results", "") for file in os.listdir(dir3) if file.endswith('.json')}
    # Find common names in all three directories
    common_names = dir1_folders.union(dir2_folders).intersection(json_files)
    common_names = {item + "_results" for item in common_names}
    return common_names


def get_top_five_files(directory, common_names):
    auc_f1_scores = []

    for name in common_names:
        file_path = os.path.join(directory, name + '.json')
        if os.path.exists(file_path):
            data = read_json(file_path)
            test_auc = data[0]['test_auc']
            test_f1 = data[0]['test_f1']
            combined_score = test_auc + test_f1
            auc_f1_scores.append((name, test_auc, test_f1, combined_score))

    # Sort and get top five files based on AUC
    top_five_files = sorted(auc_f1_scores, key=lambda x: x[3], reverse=True)[:4]
    return top_five_files


def create_scatter_plot(top_files):
    plt.figure(figsize=(8, 6))

    merged_points = {}
    max_combined_score = 0
    best_point = None

    for file_name, test_auc, test_f1, combined_score in top_files:
        label = extract_info_from_filename(file_name)
        point = (round(test_auc, 4), round(test_f1, 4))  # Round to 4 decimal places
        if combined_score > max_combined_score:
            max_combined_score = combined_score
            best_point = point
        if point in merged_points:
            merged_points[point].append(label)  # Append label if point already exists
        else:
            merged_points[point] = [label]  # Create new entry for new point

    x_values = [point[0] for point in merged_points.keys()]
    y_values = [point[1] for point in merged_points.keys()]

    margin = 0.05  # Margin for xlim and ylim
    circle_size = 100  # Circle size
    text_offset = 0.003  # Offset for text labels
    for point, labels in merged_points.items():
        combined_label = " &\n".join(labels)
        if point == best_point:
            # Plot the best point with a star marker
            plt.scatter(point[0], point[1], s=circle_size, marker='*', c='red', label=combined_label)
        else:
            # Plot other points with circle markers
            plt.scatter(point[0], point[1], s=circle_size, label=combined_label)

        plt.text(point[0] - 5*text_offset, point[1], combined_label, fontsize=9)

    # Plot the best point with a star marker
    plt.xlabel('Test AUC')
    plt.ylabel('Test F1 Score')
    plt.xlim(min(x_values) - margin, max(x_values) + margin)
    plt.ylim(min(y_values) - margin, max(y_values) + margin)

    plt.title(f'Top 5 Configurations Based on {dataset} Test AUC')

    # Add legend inside the plot
    plt.legend(loc='upper left', fontsize=8)  # Adjust location as needed

    plt.tight_layout()
    plt.show()


# Usage
dir1 = './cae_checkpoints/best-training'
dir2 = './cae_checkpoints/fastest-training'
dir3 = './cae_checkpoints/srinivasan'

common_names = get_common_files(dir1, dir2, dir3)
# dataset = "OCT-500"
dataset = "Srinivasan"
top_files = get_top_five_files(dir3, common_names)
create_scatter_plot(top_files)
