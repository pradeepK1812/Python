from stats_visualizer.convergence import plot_t_convergence
import os


def test_convergence_runs():
    expected_file = "test_plot.png"
    result = plot_t_convergence(output_path=expected_file)
    assert result == expected_file

