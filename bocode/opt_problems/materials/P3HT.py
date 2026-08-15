"""P3HT/CNT composite electrical-conductivity optimization.

Sources:
D. Bash, Y. Cai, V. Chellappan, et al. Multi-fidelity high-throughput optimization of electrical conductivity in P3HT-CNT composites. Advanced Functional Materials 31:2102606, 2021.
Dataset via the PV-Lab benchmarking suite: https://github.com/PV-Lab/Benchmarking
"""

from ._dataset_problem import MaterialsDatasetProblem


class P3HT(MaterialsDatasetProblem):
    """Maximise measured conductivity of a P3HT-CNT composite (5 content inputs)."""

    available_dimensions = 5
    csv_name = "P3HT_dataset.csv"
    feature_columns = [
        "P3HT content (%)",
        "D1 content (%)",
        "D2 content (%)",
        "D6 content (%)",
        "D8 content (%)",
    ]
    objective_column = "Conductivity (measured) (S/cm)"
    minimize = False
