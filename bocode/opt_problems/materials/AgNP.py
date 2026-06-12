"""Silver-nanoparticle synthesis optimization (AgNP).

Sources:
F. Mekki-Berrada, Z. Ren, T. Huang, et al. Two-step machine learning enables optimized nanoparticle synthesis. npj Computational Materials 7:55, 2021.
Dataset via the PV-Lab benchmarking suite: https://github.com/PV-Lab/Benchmarking
"""

from ._dataset_problem import MaterialsDatasetProblem


class AgNP(MaterialsDatasetProblem):
    """Minimise the spectral loss of a silver-nanoparticle synthesis (5 flow inputs)."""

    available_dimensions = 5
    csv_name = "AgNP_dataset.csv"
    feature_columns = ["QAgNO3(%)", "Qpva(%)", "Qtsc(%)", "Qseed(%)", "Qtot(uL/min)"]
    objective_column = "loss"
    minimize = True
