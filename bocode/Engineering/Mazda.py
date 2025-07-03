import os
import stat
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..base import BenchmarkProblem, DataType


class Mazda_SCA(BenchmarkProblem):
    """
    https://ladse.eng.isas.jaxa.jp/benchmark/
    """

    available_dimensions = 148
    input_type = DataType.CONTINUOUS
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

        # Write the DataFrame to a text file with space-separated values
        output_file_path = Path(__file__).parent / "Mazda_Data" / "pop_vars_eval.txt"

        dataframe_back.to_csv(output_file_path, sep="\t", header=False, index=False)

        #####################
        # Run Bash file
        #####################

        script_dir = Path(__file__).parent
        bin_path = script_dir / "Mazda_Data" / "bin" / "mazda_mop_sca"
        input_dir = script_dir / "Mazda_Data"

        if not os.access(bin_path, os.X_OK):
            print(f"Adding execution permissions to: {bin_path}")
            os.chmod(
                bin_path,
                os.stat(bin_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )

        # MUST BE ON A LINUX/UNIX MACHINE
        subprocess.run(
            [str(bin_path), str(input_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        #####################
        # Read in objective and constraints
        #####################

        # Read the data from the file into a pandas DataFrame
        file_path = script_dir / "Mazda_Data" / "pop_objs_eval.txt"

        objs_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        objs_data_numpy = objs_dataframe.values

        # Convert the numpy array to a torch tensor
        objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)
        # objs_data_tensor = objs_data_tensor[:,0].reshape(objs_data_tensor.shape[0],1)
        objs_data_tensor = objs_data_tensor

        # Read the data from the file into a pandas DataFrame
        file_path = script_dir / "Mazda_Data" / "pop_cons_eval.txt"
        cons_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        cons_data_numpy = cons_dataframe.values

        # Convert the numpy array to a torch tensor
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
    """

    available_dimensions = 222
    input_type = DataType.CONTINUOUS
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

        # Write the DataFrame to a text file with space-separated values
        output_file_path = Path(__file__).parent / "Mazda_Data" / "pop_vars_eval.txt"

        dataframe_back.to_csv(output_file_path, sep="\t", header=False, index=False)

        #####################
        # Run Bash file
        #####################

        script_dir = Path(__file__).parent
        bin_path = script_dir / "Mazda_Data" / "bin" / "mazda_mop"
        input_dir = script_dir / "Mazda_Data"

        if not os.access(bin_path, os.X_OK):
            print(f"Adding execution permissions to: {bin_path}")
            os.chmod(
                bin_path,
                os.stat(bin_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )

        # MUST BE ON A LINUX/UNIX MACHINE
        subprocess.run(
            [str(bin_path), str(input_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        #####################
        # Read in objective and constraints
        #####################

        # Read the data from the file into a pandas DataFrame
        file_path = script_dir / "Mazda_Data" / "pop_objs_eval.txt"
        objs_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        objs_data_numpy = objs_dataframe.values

        # Convert the numpy array to a torch tensor
        objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)
        # objs_data_tensor = objs_data_tensor[:,0].reshape(objs_data_tensor.shape[0],1)
        objs_data_tensor = objs_data_tensor

        # Read the data from the file into a pandas DataFrame
        file_path = script_dir / "Mazda_Data" / "pop_cons_eval.txt"
        cons_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        cons_data_numpy = cons_dataframe.values

        # Convert the numpy array to a torch tensor
        cons_data_tensor = torch.tensor(cons_data_numpy, dtype=torch.float32)

        return cons_data_tensor, -objs_data_tensor
