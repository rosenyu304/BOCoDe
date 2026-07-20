import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..._fetch import fetch_data_file
from ...base import BenchmarkProblem


def _run_mazda_binary(bin_path, src_data_dir, vars_df):
    """Evaluate the Mazda crash-model binary in a private temp directory.

    The binary communicates through fixed filenames inside the directory it is
    given: ``pop_vars_eval.txt`` (input) and ``pop_objs_eval.txt`` /
    ``pop_cons_eval.txt`` (output). Running each evaluation in its own temp copy
    of ``Mazda_Data`` stops concurrent jobs from clobbering each other's I/O
    files -- the shared fixed paths under ``Mazda_Data`` silently corrupted
    results otherwise. The whole data dir is copied so the binary finds every
    file it may need relative to its input directory.
    """
    with tempfile.TemporaryDirectory(prefix="mazda_") as work:
        data_dir = Path(work) / "Mazda_Data"
        shutil.copytree(src_data_dir, data_dir)
        vars_df.to_csv(
            data_dir / "pop_vars_eval.txt", sep="\t", header=False, index=False
        )
        subprocess.run(
            [str(bin_path), str(data_dir)],
            capture_output=True,
            start_new_session=True,
        )
        objs = pd.read_csv(data_dir / "pop_objs_eval.txt", sep=r"\s+", header=None).values
        cons = pd.read_csv(data_dir / "pop_cons_eval.txt", sep=r"\s+", header=None).values
        return objs, cons


class Mazda_SCA(BenchmarkProblem):
    """
    https://ladse.eng.isas.jaxa.jp/benchmark/

    Reference point (DERIVED, not published): the Mazda benchmark publishes no
    hypervolume reference point and no approximated ideal/nadir point. Derived with
    the standard convention (Tanabe & Ishibuchi, Sec. 3.1; Picard & Schiffmann,
    Sec. V-A) ``r = z_ideal + 1.1 * (z_nadir - z_ideal)`` in the minimization
    frame, where ``z_ideal`` / ``z_nadir`` are the min / max over the non-dominated
    set of a fixed Latin-hypercube sample (``problem.sample(2048, seed=0)``, all
    points, feasible or not), then negated for BoCoDe's maximization frame. Derived
    values (minimization frame): ``z_ideal = (1.90409, -20.0, 0.89946, 0.92375)``,
    ``z_nadir = (2.02548, -6.0, 0.98033, 1.05704)``.
    """

    available_dimensions = 148
    num_objectives = 4
    num_constraints = 36

    # 222D objective, 54 constraints, X = n-by-222
    # 2 Cars Optimization Case

    tags = {
        "single_objective",
        "multi_objective",
        "constrained",
        "continuous",
        "222D",
        "extra_imports",
    }

    def __init__(self):
        super().__init__(
            dim=148,
            num_objectives=4,
            num_constraints=36,
            bounds=[(0, 1)] * 148,  # Scaled upon evaluation
            ref_point=[-2.03762237, 4.6, -0.98841721, -1.07037097],
        )

    def _evaluate_implementation(self, X):
        ##########################################
        # Scaling
        ##########################################

        # Define the path to your Excel file
        file_path = Path(__file__).parent / "Mazda_Data" / "Info_Mazda_CdMOBP.xlsx"

        # Read the Excel file into a DataFrame
        dataframe = pd.read_excel(file_path, sheet_name="Explain_DV_and_Const.")

        bounds = dataframe.values[2:, 3:5].astype(float)

        bounds = np.vstack((bounds[:74], bounds[-74:]))

        bounds_tensor = torch.tensor(bounds, dtype=torch.float32)

        range_bounds = bounds_tensor[:, 1] - bounds_tensor[:, 0]

        scaled_samples = X * range_bounds + bounds_tensor[:, 0]

        # Convert the torch tensor to a numpy array
        data_numpy_back = scaled_samples.numpy()

        # Create a pandas DataFrame from the numpy array
        dataframe_back = pd.DataFrame(data_numpy_back)

        #####################
        # Run Bash file
        #####################

        script_dir = Path(__file__).parent
        bin_path = Path(
            fetch_data_file(
                "mazda_mop_sca",
                local_fallback=str(script_dir / "Mazda_Data" / "bin" / "mazda_mop_sca"),
            )
        )

        if not os.access(bin_path, os.X_OK):
            print(f"Adding execution permissions to: {bin_path}")
            os.chmod(
                bin_path,
                os.stat(bin_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )

        # MUST BE ON A LINUX/UNIX MACHINE. Isolated temp dir per eval (see
        # _run_mazda_binary) so concurrent jobs cannot corrupt each other.
        objs_data_numpy, cons_data_numpy = _run_mazda_binary(
            bin_path, script_dir / "Mazda_Data", dataframe_back
        )

        objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)
        cons_data_tensor = torch.tensor(cons_data_numpy, dtype=torch.float32)

        return cons_data_tensor, -objs_data_tensor


class Mazda(BenchmarkProblem):
    """
    https://ladse.eng.isas.jaxa.jp/benchmark/

    Meanings of each objective:
    - The first column is total weight of three vehicles.
    - The second column is number of common gauge parts.
    - The third column is weight of SUV.
    - The fourth column is weight of LV.
    - The fifth column is weight of SV.

    Reference point (DERIVED, not published): the Mazda benchmark publishes no
    hypervolume reference point and no approximated ideal/nadir point. Derived with
    the standard convention (Tanabe & Ishibuchi, Sec. 3.1; Picard & Schiffmann,
    Sec. V-A) ``r = z_ideal + 1.1 * (z_nadir - z_ideal)`` in the minimization
    frame, where ``z_ideal`` / ``z_nadir`` are the min / max over the non-dominated
    set of a fixed Latin-hypercube sample (``problem.sample(2048, seed=0)``, all
    points, feasible or not), then negated for BoCoDe's maximization frame. Derived
    values (minimization frame): ``z_ideal = (2.83845, -5.0, 0.90134, 0.90752,
    0.94304)``, ``z_nadir = (2.99307, 0.0, 1.01610, 1.02938, 1.07776)``.
    """

    available_dimensions = 222
    num_objectives = 5
    num_constraints = 54

    # 222D objective, 54 constraints, X = n-by-222
    # 3 car optimization case

    tags = {
        "single_objective",
        "multi_objective",
        "constrained",
        "continuous",
        "222D",
        "extra_imports",
    }

    def __init__(self):
        super().__init__(
            dim=222,
            num_objectives=5,
            num_constraints=54,
            bounds=[(0, 1)] * 222,  # Scaled upon evaluation
            ref_point=[-3.008535, -0.5, -1.02757455, -1.04156956, -1.09123624],
        )

    def _evaluate_implementation(self, X):
        ##########################################
        # Scaling
        ##########################################

        # Define the path to your Excel file
        file_path = Path(__file__).parent / "Mazda_Data" / "Info_Mazda_CdMOBP.xlsx"

        # Read the Excel file into a DataFrame
        dataframe = pd.read_excel(file_path, sheet_name="Explain_DV_and_Const.")

        bounds = dataframe.values[2:, 3:5].astype(float)

        bounds_tensor = torch.tensor(bounds, dtype=torch.float32)

        range_bounds = bounds_tensor[:, 1] - bounds_tensor[:, 0]

        scaled_samples = X * range_bounds + bounds_tensor[:, 0]

        # Convert the torch tensor to a numpy array
        data_numpy_back = scaled_samples.numpy()

        # Create a pandas DataFrame from the numpy array
        dataframe_back = pd.DataFrame(data_numpy_back)

        #####################
        # Run Bash file
        #####################

        script_dir = Path(__file__).parent
        bin_path = Path(
            fetch_data_file(
                "mazda_mop",
                local_fallback=str(script_dir / "Mazda_Data" / "bin" / "mazda_mop"),
            )
        )

        if not os.access(bin_path, os.X_OK):
            print(f"Adding execution permissions to: {bin_path}")
            os.chmod(
                bin_path,
                os.stat(bin_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )

        # MUST BE ON A LINUX/UNIX MACHINE. Isolated temp dir per eval (see
        # _run_mazda_binary) so concurrent jobs cannot corrupt each other.
        objs_data_numpy, cons_data_numpy = _run_mazda_binary(
            bin_path, script_dir / "Mazda_Data", dataframe_back
        )

        objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)
        cons_data_tensor = torch.tensor(cons_data_numpy, dtype=torch.float32)

        return cons_data_tensor, -objs_data_tensor
