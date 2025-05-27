import torch
import pytest
from ..CEC2017 import *

benchmark_classes = [CEC2017_p1, CEC2017_p2, CEC2017_p3, CEC2017_p4, CEC2017_p5, CEC2017_p6, CEC2017_p7, CEC2017_p8, CEC2017_p9, CEC2017_p10, 
                     CEC2017_p11, CEC2017_p12, CEC2017_p13, CEC2017_p14, CEC2017_p15, CEC2017_p16, CEC2017_p17, CEC2017_p18, CEC2017_p19, CEC2017_p20, 
                     CEC2017_p21, CEC2017_p22, CEC2017_p23, CEC2017_p24, CEC2017_p25, CEC2017_p26, CEC2017_p27, CEC2017_p28, CEC2017_p29]

@pytest.mark.parametrize("benchmark", benchmark_classes)
def test_CEC2017_evaluate(benchmark):

    problem = benchmark()

    dim = problem.dim

    if type(problem.available_dimensions) == int:
        assert dim == problem.available_dimensions, f"Dimension {dim} does not match available dimensions variable: {problem.available_dimensions}"

    rand_test_points = 5 # Number of random points to test
    
    # Generate random points within constraints
    X = torch.rand((rand_test_points, dim))

    gx, fx = problem._evaluate_implementation(X)

    assert fx.shape == (rand_test_points, problem.num_objectives), f"Unexpected fx shape: {fx.shape}"
    if gx is not None and problem.num_constraints > 0:
        assert gx.shape == (rand_test_points, problem.num_constraints), f"Unexpected gx shape: {gx.shape}"

    assert len(problem.bounds) == dim, "Number of bounds does not match dimension"
    
    if problem.num_constraints == 0:
        assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

    if problem.x_opt is not None and problem.optimum is not None:
        eval_opt = problem._evaluate_implementation(torch.Tensor(problem.x_opt))[1].float()
        assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=0.05), f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"

    if type(problem.available_dimensions) == list:
        for dim in problem.available_dimensions:
            problem = benchmark(dim=dim)

            rand_test_points = 5 # Number of random points to test
            
            # Generate random points within constraints
            X = torch.rand((rand_test_points, dim))

            gx, fx = problem._evaluate_implementation(X)

            assert fx.shape == (rand_test_points, problem.num_objectives), f"Unexpected fx shape: {fx.shape}"
            if gx is not None and problem.num_constraints > 0:
                assert gx.shape == (rand_test_points, problem.num_constraints), f"Unexpected gx shape: {gx.shape}"

            assert len(problem.bounds) == dim, "Number of bounds does not match dimension"
            
            if problem.num_constraints == 0:
                assert torch.isfinite(fx).all(), "fx contains NaN or Inf values"

            if problem.x_opt is not None and problem.optimum is not None:
                eval_opt = problem._evaluate_implementation(torch.Tensor(problem.x_opt))[1].float()
                assert torch.allclose(eval_opt, torch.Tensor(problem.optimum), atol=0.05, rtol=0.01), f"X_opt ({problem.x_opt}) evaluation ({eval_opt}) does not match optimum ({problem.optimum})"

    print(f"Test passed")