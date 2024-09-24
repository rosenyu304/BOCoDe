import os
import subprocess
import sys
import tempfile
from pathlib import Path
from platform import machine

import numpy as np
import torch
import stat


def MOPTA08_Car_single(x):
    # Get the current permissions of the file
    current_permissions = os.stat(os.getcwd()).st_mode
    
    # Add execute permissions for the owner, group, and others
    new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    
    # Apply the new permissions
    os.chmod(os.getcwd(), new_permissions)
    
    sysarch = 64 if sys.maxsize > 2 ** 32 else 32
    # machine = machine().lower()
    
    # if machine == "armv7l":
    #     assert sysarch == 32, "Not supported"
    #     mopta_exectutable = "mopta08_armhf.bin"
    # elif machine == "x86_64":
    #     assert sysarch == 64, "Not supported"
    #     mopta_exectutable = "mopta08_elf64.bin"
    # elif machine == "i386":
    #     assert sysarch == 32, "Not supported"
    #     mopta_exectutable = "mopta08_elf32.bin"
    # else:
    #     raise RuntimeError("Machine with this architecture is not supported")

    machine = "x86_64"
    mopta_exectutable = "mopta08_elf64.bin"

    mopta_full_path = os.path.join(
        "mopta08", mopta_exectutable
    )
    # print(mopta_full_path)
    
    directory_file_descriptor = tempfile.TemporaryDirectory()
    # directory_name = directory_file_descriptor.name
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
        # '#!/home/rosen/Dropbox (MIT)/Rosen_DeC0De/MOPTA_Test/mopta08/mopta08_elf64.bin',
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
    # print(output)
    # print(x)
    # print(output)
    output = [x.strip() for x in output]
    output = np.array([float(x) for x in output if len(x) > 0])
    value = output[0]
    constraints = output[1:]
    # print(value, constraints)

    return constraints, value



def MOPTA08_Car(X):
    GX =  torch.zeros(X.shape[0], 68)
    FX =  torch.zeros(X.shape[0], 1)
    for ii in range(X.shape[0]):
        input_x = X[ii,:].numpy()
        gx, fx = MOPTA08_Car_single(input_x)
        GX[ii,:] = torch.from_numpy(gx)
        FX[ii,:] = -fx

    return GX, FX




def MOPTA08_Car_softpen(X):
    GX =  torch.zeros(X.shape[0], 68)
    FX =  torch.zeros(X.shape[0], 1)
    for ii in range(X.shape[0]):
        input_x = X[ii,:].numpy()
        gx, fx = MOPTA08_Car_single(input_x)
        GX[ii,:] = torch.from_numpy(gx)
        FX[ii,:] = fx

    cost = GX
    cost[cost<0] = 0
    cost = cost.sum(dim=1).reshape(cost.shape[0], 1)
    FX = FX + cost

    return GX, -FX


















