# TOPFuns

We present TOPFuns (Torch-based OPtimization Functions for engineering), a Python and PyTorch-based library that contains the most comprehensive suite of engineering design problems and an interface to popular synthetic optimization problems, with access to 300+ problems for optimization benchmarking. Our goal is to provide not only a Python optimization benchmark library but also to allow the PyTorch interface for facilitating machine learning optimization algorithms and applications such as surrogate and Bayesian optimization.

# What is in TOPFuns?

## Engineering design problems
We present a diverse collection of engineering design problems including car design, cantilever beam, truss structure optimization, and physics simulation of robotics. 

<img src="docs/Figures/TopFuns_Icon_v1.png" width="300">
<!-- ![figure_icon](docs/Figures/TopFuns_Icon_v1.png) -->

## Interface to popular benchmarks
still working on this
| Botorch  | BBOB/COCO | LassoBench |
| :------: | :------:  | :------:   |
| <img src="docs/Figures/botorch_icon.png" width="50">  | need figure      | need figure       |


# Installation

For our own testing now:
```
pip install git+https://github.com/rosenyu304/OptBenckmarkLibrary@WIP/May27
```

After PyPI upload:
```
pip install TOPFuns
```

# Optimization Problem Definition
Here we define all our problems for **maximization** optimization algorithms (therefore please negate the problem if your optimization algorithms are doing minimization). For the constraints here, they are inequality constraints with gx <= 0 as feasible.

# Example Usage

For details of each problem's usage, please read our docs. Here we provide examples to common usage of this library:

1. Direct evaluation
```
import TOPFuns
import torch

# Instantiate a Synthetic benchmark problem
problem = TOPFuns.Engineering.Car()

# Evaluate at a point
x = torch.Tensor([[0.0] * problem.dim])
constraints, values = problem._evaluate_implementation(x)

print(f"Is it feasible? {(constraints<=0).all()}")
print(f"Function value at origin: {values[0]}")
```

2. Scaling parameters sampled from unit hypercube (typical Bayesian optimization practice)
```
import TOPFuns
import torch

# Instantiate a Synthetic benchmark problem
problem = TOPFuns.Synthetics.Ackley(show_info=True) # show_info=True to show info of the problem 

# Evaluate at a in bounds of [0,1]s
x = torch.rand(5,problem.dim)
print("X in [0,1]s:\n",x,"\n")

# Scale it w.r.t. the problem bounds
x = problem.scale(x)
print("Scaled X in bounds:\n",x)
constraints, values = problem._evaluate_implementation(x)

print(f"Is each sample feasible? {(constraints<=0).all(dim=1)}")
print(f"Function value at origin: {values[0]}")
```

3. Example using a scipy minimization for this

4. Synthetic function visualization



# Development

TOPFuns is an open source projects and we welcome contributions! If you want to add a new problem, please reach out to us first to see if it is a good fit for TOPFuns.

# Citing

1. If you use TOPFuns in your research, please cite the following paper:
```
todo
```

2. If you use the the TOPFuns interfaces to other libraries or open source code functions (ex: BoTorch, BBOB, NEORL, MODAact, LassoBench, ...), please cite them accordingly.
