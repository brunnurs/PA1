import matplotlib.pyplot as plt
import numpy as np


def plot_multiple_metrics_learning_curve(metrics):

    # x-values will always be the same
    x_data_points = metrics[0].number_of_training_examples

    for idx, metric in enumerate(metrics):
        plt.plot(x_data_points, metric.f1_values, label='Run {}'.format(idx + 1), linewidth=0.5)

    y_mean = []
    for idx in range(len(x_data_points)):
        y_mean.append(np.mean([metric.f1_values[idx] for metric in metrics]))

    plt.plot(x_data_points, y_mean, label='Mean', marker='o', linewidth=2.0, color='red')

    plt.ylabel('f1 value (matches only)')
    plt.xlabel('size training set')
    plt.title('learning curve')

    plt.show()
