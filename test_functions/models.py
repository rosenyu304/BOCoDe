# Author: Qiuyi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class _Combo(nn.Module):    
    def forward(self, input):
        return self.model(input)
        
class LinearCombo(_Combo):
    def __init__(self, in_features, out_features, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            activation
        )

class SNLinearCombo(_Combo):
    def __init__(self, in_features, out_features, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(in_features, out_features)),
            activation
        )
    
class MLP(nn.Module):
    """Regular fully connected network generating features.
    
    Args:
        in_features: The number of input features.
        out_feature: The number of output features.
        layer_width: The widths of the hidden layers.
        combo: The layer combination to be stacked up.
    
    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output: `(N, H_out)` where H_out = out_features.
    """
    def __init__(
        self, in_features: int, out_features:int, layer_width: list, 
        combo = LinearCombo
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_width = list(layer_width)
        self.model = self._build_model(combo)
    
    def forward(self, input):
        return self.model(input)

    def _build_model(self, combo):
        model = nn.Sequential()
        idx = -1
        for idx, (in_ftr, out_ftr) in enumerate(self.layer_sizes[:-1]):
            model.add_module(str(idx), combo(in_ftr, out_ftr))
        model.add_module(str(idx+1), nn.Linear(*self.layer_sizes[-1])) # type:ignore
        return model
    
    @property
    def layer_sizes(self):
        return list(zip([self.in_features] + self.layer_width, 
        self.layer_width + [self.out_features]))


class SNMLP(MLP):
    def __init__(
        self, in_features: int, out_features: int, layer_width: list, 
        combo=SNLinearCombo):
        super().__init__(in_features, out_features, layer_width, combo) 
    
    def _build_model(self, combo):
        model = nn.Sequential()
        idx = -1
        for idx, (in_ftr, out_ftr) in enumerate(self.layer_sizes[:-1]):
            model.add_module(str(idx), combo(in_ftr, out_ftr))
        model.add_module(str(idx+1), spectral_norm(nn.Linear(*self.layer_sizes[-1]))) # type:ignore
        return model