import torch

def cont_to_disc(x, disc_values):
    # Convert continuous value to discrete value
    # Input:
    #   x: continuous value in [0, 1]
    #   disc_values: discrete values
    # Output: discrete value
    idx = torch.floor(x * len(disc_values)).long()
    return disc_values[torch.clamp(idx, 0, len(disc_values)-1)]