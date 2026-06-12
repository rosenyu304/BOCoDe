import bocode
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

# Initialize observed data
dim = 11
train_X = torch.rand(20, dim)

# Problem setup
problem = bocode.Car()
train_objectives, train_constraints = problem.evaluate(train_X.clone())
penalty = torch.clamp(train_constraints, min=0).sum(dim=1, keepdim=True)
train_Y = train_objectives - penalty

# Example code from botorch getting started

for i in range(10):  # Optimization loop for 10 iterations
    # Set up GP model
    gp = SingleTaskGP(
        train_X=train_X.clone().to(torch.double),
        train_Y=train_Y.clone().to(torch.double),
        input_transform=Normalize(d=dim),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Set up acquisition function
    logNEI = LogExpectedImprovement(model=gp, best_f=train_Y.max())

    # Set up bounds
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(torch.double)

    # Optimize acquisition function and get next point
    candidate, acq_value = optimize_acqf(
        logNEI,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )

    # Append new point to training data and evaluate it
    train_X = torch.cat([train_X, candidate], dim=0)
    train_objectives, train_constraints = problem.evaluate(train_X.clone())
    penalty = torch.clamp(train_constraints, min=0).sum(dim=1, keepdim=True)
    train_Y = train_objectives - penalty

    print(f"Iteration {i + 1}: Best f = {-train_Y.max().item()}")
