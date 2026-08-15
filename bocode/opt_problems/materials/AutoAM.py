"""Autonomous additive-manufacturing (3D-printing) process optimization.

Sources:
J. R. Deneault, J. Chang, J. Myung, et al. Toward autonomous additive manufacturing: Bayesian optimization on a 3D printer. MRS Bulletin 46:566-575, 2021.
Dataset via the PV-Lab benchmarking suite: https://github.com/PV-Lab/Benchmarking
"""

from ._dataset_problem import MaterialsDatasetProblem


class AutoAM(MaterialsDatasetProblem):
    """Maximise the print-quality score of a 3D printer (4 process inputs)."""

    available_dimensions = 4
    csv_name = "AutoAM_dataset.csv"
    feature_columns = [
        "Prime Delay",
        "Print Speed",
        "X Offset Correction",
        "Y Offset Correction",
    ]
    objective_column = "Score"
    minimize = False
