from stats_visualizer.convergence import plot_t_to_normal_convergence
import os


def test_convergence_runs():
    """Verify plotting executes and saves output."""
    output_path = "test_plot.png"
    result = plot_t_to_normal_convergence(save_path=output_path)

    assert result == output_path
    assert os.path.exists(output_path)

    os.remove(output_path)
