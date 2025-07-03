import os
import platform
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ..base import BenchmarkProblem, DataType


class MOPTA08Car(BenchmarkProblem):
    """
    https://leonard.papenmeier.io/2023/02/09/mopta08-executables.html
    """

    available_dimensions = 124
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 68

    # 124D objective, 68 constraints, X = n-by-124

    tags = {"single_objective", "constrained", "continuous", "124D", "extra_imports"}

    def __init__(self):
        super().__init__(
            dim=124,
            num_objectives=1,
            num_constraints=68,
            bounds=[(0, 1)] * 124,
            optimum=[222.74],
        )

    def _evaluate_implementation(self, X):
        def MOPTA08_Car_single(x):
            machine = platform.machine().lower()
            sysarch = 64 if sys.maxsize > 2**32 else 32

            if machine == "armv7l":
                assert sysarch == 32, "Not supported"
                mopta_exectutable = "mopta08_armhf.bin"
            elif machine == "x86_64":
                assert sysarch == 64, "Not supported"
                mopta_exectutable = "mopta08_elf64.bin"
            elif machine == "i386":
                assert sysarch == 32, "Not supported"
                mopta_exectutable = "mopta08_elf32.bin"
            else:
                raise RuntimeError("Machine with this architecture is not supported")

            # Add execute permissions for the owner, group, and others
            script_dir = Path(__file__).parent
            mopta_full_path = script_dir / "Mopta_Data" / mopta_exectutable
            mopta_full_path = os.path.join(mopta_full_path)

            if not os.access(mopta_full_path, os.X_OK):
                print(f"Adding execution permissions to: {mopta_full_path}")
                os.chmod(
                    mopta_full_path,
                    os.stat(mopta_full_path).st_mode
                    | stat.S_IXUSR
                    | stat.S_IXGRP
                    | stat.S_IXOTH,
                )

            sysarch = 64 if sys.maxsize > 2**32 else 32

            directory_name = Path(__file__).parent / "Mopta_Data"

            with open(os.path.join(directory_name, "input.txt"), "w+") as tmp_file:
                for _x in x:
                    tmp_file.write(f"{_x}\n")

            _ = subprocess.run(
                mopta_full_path,
                stdout=subprocess.PIPE,
                cwd=directory_name,
                shell=True,
            )

            with open(os.path.join(directory_name, "output.txt"), "r") as tmp_file:
                tmp_file.seek(0)
                output = tmp_file.read().split("\n")
            output = [m.strip() for m in output]
            output = output[:-1]
            output = np.array([float(m) for m in output if len(x) > 0])
            value = output[0]
            constraints = output[1:]

            return constraints, value

        fx = np.zeros((X.shape[0], 1))
        gx = np.zeros((X.shape[0], 68))

        for i in range(X.shape[0]):
            # Get objectives and constraints for each row
            gx[i], fx[i] = MOPTA08_Car_single(X[i, :].numpy())

        return torch.from_numpy(gx), torch.from_numpy(fx)
