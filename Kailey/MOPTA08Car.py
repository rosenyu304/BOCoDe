import torch
from base import BenchmarkProblem

class MOPTA08Car(BenchmarkProblem):

    r'''

    '''

    # 124D objective, 68 constraints, X = n-by-124

    tags = {"single_objective", "constrained", "continuous", "124D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 124, num_obj = 1, num_cons = 68, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        n = X.size(0)

        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from platform import machine

        import numpy as np
        import stat

        def MOPTA08_Car_single(x):
            # Get the current permissions of the file
            current_permissions = os.stat(os.getcwd()).st_mode

            # Add execute permissions for the owner, group, and others
            new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

            # Apply the new permissions
            os.chmod(os.getcwd(), new_permissions)

            sysarch = 64 if sys.maxsize > 2 ** 32 else 32

            machine = "x86_64"
            mopta_exectutable = "mopta08_elf64.bin"

            mopta_full_path = os.path.join(
                "mopta08", mopta_exectutable
            )

            directory_file_descriptor = tempfile.TemporaryDirectory()
            directory_name = Path(__file__).parent

            ##########################################################################################
            # Input here
            # if x == None:
            #     x = np.random.rand(124)
            #     print(x.shape)
            ##########################################################################################
            with open(os.path.join(directory_name, "input.txt"), "w+") as tmp_file:
                for _x in x:
                    tmp_file.write(f"{_x}\n")
            popen = subprocess.Popen(
                mopta_full_path,
                stdout=subprocess.PIPE,
                cwd=directory_name,
                shell=True,
            )
            popen.wait()

            with open(os.path.join(directory_name, "output.txt"), "r") as  tmp_file:
                output = (
                    tmp_file
                    .read()
                    .split("\n")
                )
            output = [x.strip() for x in output]
            output = np.array([float(x) for x in output if len(x) > 0])
            output = np.array(x)
            value = output[0]
            constraints = output[1:]

            return constraints, value


        # GX =  torch.zeros(n, 68)
        # FX =  torch.zeros(n, 1)
        # for ii in range(n):
        # input_x = X[ii,:].numpy()
        gx, fx = MOPTA08_Car_single(X)
            # GX[ii,:] = torch.from_numpy(gx)
            # FX[ii,:] = -fx
        return torch.from_numpy(gx), torch.from_numpy(fx)
        # return GX, FX
