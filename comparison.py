import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os


def extract_tensorboard_data(log_dir):
    # Initialize an accumulator to extract scalar data
    ea = event_accumulator.EventAccumulator(log_dir,
                                            size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()  # Load the data from the log directory

    # Assuming you're interested in scalar data
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = [event.value for event in events]
    return data

import matplotlib.pyplot as plt
import itertools


import matplotlib.pyplot as plt
import itertools

def plot_and_analyze_metrics(all_data, metric, loss_type, top_n=5):
    """
    Plots the specified metric for all models and prints the top_n models
    based on both the metric and steps.

    :param all_data: Dictionary containing model data.
    :param metric: String representing the metric to plot ('train_loss', 'val_loss_epoch', or 'val_loss_step').
    :param loss_type: String representing the type of loss.
    :param top_n: Integer representing the number of top models to display.
    """

    color_cycle = itertools.cycle(plt.cm.tab20.colors)

    # Lists to store metric value and steps info
    metric_values_info = []
    steps_info = []

    plt.figure(figsize=(12, 7))
    for log_dir, data in all_data.items():
        config = log_dir.split(loss_type)[1]
        optimizer = config.split("_")[1]
        scheduler = config.split("_")[2].split("/")[0]
        label = f'{optimizer} - {scheduler}'

        if metric in data:
            step_range = range(len(data[metric]))
            plt.plot(step_range, data[metric],
                     label=label,
                     color=next(color_cycle), linewidth=1.5)

            min_metric_value = min(data[metric])
            metric_values_info.append((min_metric_value, len(step_range), label))
            steps_info.append((len(step_range), min_metric_value, label))

    plt.xlabel('Step', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{metric.replace("_", " ").title()} Across Models using {loss_type.upper()} Loss', fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

    # Sort and extract top_n models
    metric_values_info.sort()
    steps_info.sort()

    top_models_by_metric = metric_values_info[:top_n]
    top_models_by_steps = steps_info[:top_n]

    # Print the results
    print(f"Top {top_n} models with minimum {metric} and their corresponding steps:")
    for value, steps, label in top_models_by_metric:
        print(f"{label}: {metric.title()} = {value}, Steps = {steps}")

    print(f"\nTop {top_n} models with minimum steps and their corresponding {metric}:")
    for steps, value, label in top_models_by_steps:
        print(f"{label}: Steps = {steps}, {metric.title()} = {value}")
    print("=============================================================")

# Example usage
loss_type = "mse"
root = "./checkpoints"
log_dirs = glob.glob(root + f"/**{loss_type}**")
log_dirs = [p + "/tb_log/mce/version_0" for p in log_dirs]
all_data = {log_dir: extract_tensorboard_data(log_dir) for log_dir in log_dirs}

# Example usage
plot_and_analyze_metrics(all_data, 'train_loss', loss_type)
plot_and_analyze_metrics(all_data, 'val_loss_epoch', loss_type)
plot_and_analyze_metrics(all_data, 'val_loss_step', loss_type)
