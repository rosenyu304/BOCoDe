import torch
from base import BenchmarkProblem

class Mazda(BenchmarkProblem):

    r'''
    https://ladse.eng.isas.jaxa.jp/benchmark/
    '''

    # 222D objective, 54 constraints, X = n-by-222

    tags = {"single_objective", "multi_objective", "constrained", "continuous", "222D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 222, num_obj = 1, num_cons = 68, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        import os
        import subprocess
        import stat
        import pandas as pd

        ##########################################
        # Scaling
        ##########################################

        # Define the path to your Excel file
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/Info_Mazda_CdMOBP_edited.xlsx'

        # Read the Excel file into a DataFrame
        dataframe = pd.read_excel(file_path, sheet_name='Explain_DV_and_Const.')

        # Display the DataFrame to ensure it has been read correctly
        bounds = dataframe.values[1:, 1:3]
        bounds_tensor = torch.tensor(bounds, dtype=torch.float32)
        # print(bounds_tensor.shape)

        range_bounds = bounds_tensor[:,1] - bounds_tensor[:,0]

        scaled_samples = X * range_bounds + bounds_tensor[:,0]
        # print(scaled_samples)

        # Convert the torch tensor to a numpy array
        data_numpy_back = scaled_samples.numpy()

        # Create a pandas DataFrame from the numpy array
        dataframe_back = pd.DataFrame(data_numpy_back)

        # Write the DataFrame to a text file with space-separated values
        output_file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_vars_eval.txt'

        dataframe_back.to_csv(output_file_path, sep='\t', header=False, index=False)
        #####################
        #####################


        #####################
        # Run Bash file
        #####################

        # Change the current working directory
        os.chdir('/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2')

        # Get the current permissions of the file
        current_permissions = os.stat(os.getcwd()).st_mode

        # Add execute permissions for the owner, group, and others
        new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

        # Apply the new permissions
        os.chmod(os.getcwd(), new_permissions)

        # Script name
        script_name = 'run.sh'

        # Run the bash script in the background
        process = subprocess.Popen(['bash', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        process.wait()

        # Optional: capture the output and error messages
        stdout, stderr = process.communicate()

        os.chdir('/home/turbo/rosenyu/Bank_High_DIM/')
        # print(os.getcwd())
        #####################
        #####################


        #####################
        # Read in objective and constraints
        #####################

        # Read the data from the file into a pandas DataFrame
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_objs_eval.txt'
        objs_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        objs_data_numpy = objs_dataframe.values

        # Convert the numpy array to a torch tensor
        objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)
        objs_data_tensor = objs_data_tensor[:,0].reshape(objs_data_tensor.shape[0],1)

        # Read the data from the file into a pandas DataFrame
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_cons_eval.txt'
        cons_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        cons_data_numpy = cons_dataframe.values

        # Convert the numpy array to a torch tensor
        cons_data_tensor = torch.tensor(cons_data_numpy, dtype=torch.float32)

        return cons_data_tensor, -objs_data_tensor


class Mazda_softpen(BenchmarkProblem):

    r'''
    https://ladse.eng.isas.jaxa.jp/benchmark/
    '''

    # 222D objective, 54 constraints, X = n-by-222

    tags = {"single_objective", "multi_objective", "constrained", "continuous", "222D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 222, num_obj = 1, num_cons = 68, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        import os
        import subprocess
        import stat
        import pandas as pd

        ##########################################
        # Scaling
        ##########################################

        # Define the path to your Excel file
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/Info_Mazda_CdMOBP_edited.xlsx'

        # Read the Excel file into a DataFrame
        dataframe = pd.read_excel(file_path, sheet_name='Explain_DV_and_Const.')

        # Display the DataFrame to ensure it has been read correctly
        bounds = dataframe.values[1:, 1:3]
        bounds_tensor = torch.tensor(bounds, dtype=torch.float32)
        # print(bounds_tensor.shape)

        range_bounds = bounds_tensor[:,1] - bounds_tensor[:,0]

        bounds_tensor = bounds_tensor.to("cpu")
        range_bounds = range_bounds.to("cpu")
        init_samples = init_samples.to("cpu")

        scaled_samples = init_samples * range_bounds + bounds_tensor[:,0]
        # print(scaled_samples)

        # Convert the torch tensor to a numpy array
        data_numpy_back = scaled_samples.numpy()

        # Create a pandas DataFrame from the numpy array
        dataframe_back = pd.DataFrame(data_numpy_back)

        # Write the DataFrame to a text file with space-separated values
        output_file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_vars_eval.txt'

        dataframe_back.to_csv(output_file_path, sep='\t', header=False, index=False)
        #####################
        #####################


        #####################
        # Run Bash file
        #####################

        # Change the current working directory
        os.chdir('/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2')

        # Get the current permissions of the file
        current_permissions = os.stat(os.getcwd()).st_mode

        # Add execute permissions for the owner, group, and others
        new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

        # Apply the new permissions
        os.chmod(os.getcwd(), new_permissions)

        # Script name
        script_name = 'run.sh'

        # Run the bash script in the background
        process = subprocess.Popen(['bash', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        process.wait()

        # Optional: capture the output and error messages
        stdout, stderr = process.communicate()

        os.chdir('/home/turbo/rosenyu/Bank_High_DIM/')
        # print(os.getcwd())
        #####################
        #####################


        #####################
        # Read in objective and constraints
        #####################

        # Read the data from the file into a pandas DataFrame
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_objs_eval.txt'
        objs_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        objs_data_numpy = objs_dataframe.values

        # Convert the numpy array to a torch tensor
        objs_data_tensor = torch.tensor(objs_data_numpy, dtype=torch.float32)
        objs_data_tensor = objs_data_tensor[:,0].reshape(objs_data_tensor.shape[0],1)

        # Read the data from the file into a pandas DataFrame
        file_path = '/home/turbo/rosenyu/Bank_High_DIM/Mazda_CdMOBP/Mazda_CdMOBP/rosen_sample_t2/pop_cons_eval.txt'
        cons_dataframe = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Convert the DataFrame to a numpy array
        cons_data_numpy = cons_dataframe.values

        # Convert the numpy array to a torch tensor
        cons_data_tensor = torch.tensor(cons_data_numpy, dtype=torch.float32)

        cost = cons_data_tensor
        cost[cost<0] = 0
        cost = cost.sum(dim=1).reshape(cost.shape[0], 1)
        objs_data_tensor = objs_data_tensor + cost

        return cons_data_tensor, -objs_data_tensor
