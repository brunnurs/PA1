from unittest import TestCase

from active_learning.metrics import Metrics
from active_learning.multiple_metrics_plotter import plot_multiple_metrics_learning_curve
from active_learning.oracle import Oracle


class TestMultipleMetricsPlotter(TestCase):
    def test_plot_multiple_metrics_learning_curve(self):
        m1 = Metrics(Oracle([]))
        m1.number_of_training_examples.append(10)
        m1.f1_values.append(0.2)
        m1.number_of_training_examples.append(20)
        m1.f1_values.append(0.3)
        m1.number_of_training_examples.append(30)
        m1.f1_values.append(0.4)

        m2 = Metrics(Oracle([]))
        m2.number_of_training_examples.append(10)
        m2.f1_values.append(0.4)
        m2.number_of_training_examples.append(20)
        m2.f1_values.append(0.6)
        m2.number_of_training_examples.append(30)
        m2.f1_values.append(0.8)

        plot_multiple_metrics_learning_curve([m1, m2])
