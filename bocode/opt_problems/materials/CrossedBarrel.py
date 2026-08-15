"""Crossed-barrel mechanical design optimization.

Sources:
A. E. Gongora, B. Xu, W. Perry, et al. A Bayesian experimental autonomous researcher for mechanical design. Science Advances 6(15):eaaz1708, 2020.
Dataset via the PV-Lab benchmarking suite: https://github.com/PV-Lab/Benchmarking
"""

from ._dataset_problem import MaterialsDatasetProblem


class CrossedBarrel(MaterialsDatasetProblem):
    """Maximise toughness of a crossed-barrel structure (4 discrete geometric factors).

    The experiment is a 4-factor discrete design, not a continuous box: the number
    of hollow pillars ``n`` (4 levels), their twist angle ``theta`` (9 levels), their
    radius ``r`` (11 levels) and the thickness of the outer shell ``t`` (3 levels).
    600 of the 4x9x11x3 = 1188 factor combinations were fabricated, each measured in
    triplicate (1800 rows); the candidate pool holds the 600 unique combinations with
    the toughness averaged over each triplicate.
    """

    available_dimensions = 4
    csv_name = "Crossed barrel_dataset.csv"
    feature_columns = ["n", "theta", "r", "t"]
    objective_column = "toughness"
    minimize = False

    def __init__(self) -> None:
        super().__init__()
        # Ordinal factors: the allowed levels are the measured levels of each column.
        self.variable_types = [
            sorted({float(v) for v in self._X[:, i].tolist()})
            for i in range(self._X.shape[1])
        ]
