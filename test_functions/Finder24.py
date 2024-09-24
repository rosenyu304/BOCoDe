import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from datasets import *
from models import MLP, SNMLP
from torch.quasirandom import SobolEngine
import time
import sys

def LV_embedding(X, iters):
    X = X.clone()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    print(device)
    ambient_dim = X.size(-1)

    width = ambient_dim * 16
    # Note in particular the lack of the bottleneck choice below
    encoder = MLP(ambient_dim, ambient_dim, [width] * 4).to(device)
    # Note also the change in the decoder to have spectral normalization
    decoder = SNMLP(ambient_dim, ambient_dim, [width] * 4).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    
    η, λ = 0.01, 0.03

    # START_TIME = time.time()
    for i in range(iters):
        opt.zero_grad()
        z = encoder(X)
        rec_loss = F.mse_loss(decoder(z), X)
        # Note below the least volume loss
        vol_loss = torch.exp(torch.log(z.std(0) + η).mean())
        loss = rec_loss + λ * vol_loss
        loss.backward()
        opt.step()

        if (i+1) % 1000 == 0:
            print('Epoch {}: rec = {}, vol = {}'.format(i, rec_loss, vol_loss))
            
    
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        z = encoder(X)
    idx = z.std(0).argsort(descending=True)
    
    return z.to('cpu'), z.std(0).to('cpu'), idx.to('cpu'), encoder.to('cpu'), decoder.to('cpu')
    

def LV_embedding_AdamW(X, iters):
    X = X.clone()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    print(device)
    ambient_dim = X.size(-1)

    width = ambient_dim * 16
    # Note in particular the lack of the bottleneck choice below
    encoder = MLP(ambient_dim, ambient_dim, [width] * 4).to(device)
    # Note also the change in the decoder to have spectral normalization
    decoder = SNMLP(ambient_dim, ambient_dim, [width] * 4).to(device)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    
    η, λ = 0.01, 0.03
    
    START_TIME = time.time()

    for i in range(iters):
        opt.zero_grad()
        z = encoder(X)
        rec_loss = F.mse_loss(decoder(z), X)
        # Note below the least volume loss
        vol_loss = torch.exp(torch.log(z.std(0) + η).mean())
        loss = rec_loss + λ * vol_loss
        loss.backward()
        opt.step()

        if (i+1) % 1000 == 0:
            print('Epoch {}: rec = {}, vol = {}'.format(i, rec_loss, vol_loss))
            END_TIME = time.time()
            INTERVAL = END_TIME-START_TIME
            f = open("/home/gridsan/ryu/Bank_High_DIM/LVAE_Test_July9/LV_embedding_AdamW_July9.txt", "a")
            f.write('Epoch {}: rec = {}, vol = {}, Time: {}'.format(i, rec_loss, vol_loss, INTERVAL))
            f.write('\n')
            f.close()
    
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        z = encoder(X)
    idx = z.std(0).argsort(descending=True)
    
    return z.to('cpu'), z.std(0).to('cpu'), idx.to('cpu'), encoder.to('cpu'), decoder.to('cpu')    
    
    
def LV_embedding_5e4(X, iters):
    X = X.clone()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    print(device)
    ambient_dim = X.size(-1)

    width = ambient_dim * 16
    # Note in particular the lack of the bottleneck choice below
    encoder = MLP(ambient_dim, ambient_dim, [width] * 4).to(device)
    # Note also the change in the decoder to have spectral normalization
    decoder = SNMLP(ambient_dim, ambient_dim, [width] * 4).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-4)
    
    η, λ = 0.01, 0.03
    
    START_TIME = time.time()

    for i in range(iters):
        opt.zero_grad()
        z = encoder(X)
        rec_loss = F.mse_loss(decoder(z), X)
        # Note below the least volume loss
        vol_loss = torch.exp(torch.log(z.std(0) + η).mean())
        loss = rec_loss + λ * vol_loss
        loss.backward()
        opt.step()

        if (i+1) % 1000 == 0:
            print('Epoch {}: rec = {}, vol = {}'.format(i, rec_loss, vol_loss))
            END_TIME = time.time()
            INTERVAL = END_TIME-START_TIME
            f = open("/home/gridsan/ryu/Bank_High_DIM/LVAE_Test_July9/LV_embedding_5e4_July9.txt", "a")
            f.write('Epoch {}: rec = {}, vol = {}, Time: {}'.format(i, rec_loss, vol_loss, INTERVAL))
            f.write('\n')
            f.close()
    
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        z = encoder(X)
    idx = z.std(0).argsort(descending=True)
    
    return z.to('cpu'), z.std(0).to('cpu'), idx.to('cpu'), encoder.to('cpu'), decoder.to('cpu')     
    

def sampling_z(z, n_candidate=None):
    z_dim = z.shape[1]
    
    sobol = SobolEngine(z_dim, scramble=True)

    if n_candidate==None:
        n_candidate = 2000
        
    Z_samples = sobol.draw(n_candidate)
    
    for ii in range(z_dim):
        Z_samples[:,ii] = Z_samples[:,ii] * (z[:,ii].max() - z[:,ii].min()) + z[:,ii].min()
    
    return Z_samples





