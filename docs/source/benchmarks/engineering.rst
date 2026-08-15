.. _engineering_benchmarks:

Engineering Benchmarks
=================

The Engineering benchmark collection contains various Engineering-related functions.

Available Problems
----------------


* :code:`bocode.CarSideImpact`
* :code:`bocode.EulerBernoulliBeamBending`
* :code:`bocode.GearTrain`
* :code:`bocode.Mazda_SCA`
* :code:`bocode.Mazda`
* :code:`bocode.MOPTA08Car`
* :code:`bocode.RobotPush`
* :code:`bocode.Rover`
* :code:`bocode.Truss10D`
* :code:`bocode.Truss25D`
* :code:`bocode.TwoBarTruss`
* :code:`bocode.WaterProblem`
* :code:`bocode.WaterResources`
* Bayesian CHT Functions (:code:`bocode.BayesianCHT`) 
    `Source <https://link.springer.com/article/10.1007/s00158-024-03859-y>`_

     Y.-K. Tsai and R. J. Malak Jr, “Surrogate-assisted constraint-handling technique for constrained parametric multi-objective optimization,” Structural and Multidisciplinary Optimization, 2024. 
    
    * :code:`bocode.BayesianCHT.NonLinearConstraintProblemA3`
    * :code:`bocode.BayesianCHT.NonLinearConstraintProblemA4`
    * :code:`bocode.BayesianCHT.NonLinearConstraintProblemA7`
    * :code:`bocode.BayesianCHT.NonLinearConstraintProblemA8`
    * :code:`bocode.BayesianCHT.NonLinearConstraintProblemB3`
    * :code:`bocode.BayesianCHT.NonLinearConstraintProblemB4`
    * :code:`bocode.BayesianCHT.NonLinearConstraintProblemB7`
    * :code:`bocode.BayesianCHT.NonLinearConstraintProblemB8`
* Mujoco Functions (:code:`bocode.Gym`) 
    `Source <https://gymnasium.farama.org/environments/mujoco/>`_

     E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., pp. 5026–5033, 2012. doi: 10.1109/IROS.2012.6386109.

    * :code:`bocode.Gym.AntProblem`
    * :code:`bocode.Gym.HalfCheetahProblem`
    * :code:`bocode.Gym.HopperProblem`
    * :code:`bocode.Gym.HumanoidProblem`
    * :code:`bocode.Gym.HumanoidStandupProblem`
    * :code:`bocode.Gym.InvertedDoublePendulumProblem`
    * :code:`bocode.Gym.InvertedPendulumProblem`
    * :code:`bocode.Gym.PusherProblem`
    * :code:`bocode.Gym.ReacherProblem`
    * :code:`bocode.Gym.SwimmerProblem`
    * :code:`bocode.Gym.Walker2DProblem`
    * :code:`bocode.Gym.SwimmerPolicySearchProblem`
    * :code:`bocode.Gym.AntPolicySearchProblem`
    * :code:`bocode.Gym.HalfCheetahPolicySearchProblem`
    * :code:`bocode.Gym.HopperPolicySearchProblem`
    * :code:`bocode.Gym.Walker2DPolicySearchProblem`

Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Create a Botorch benchmark problem
    problem = bocode.GearTrain()
    
    # Get problem information
    bounds = problem.bounds
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    values, constraints = problem.evaluate(x)
    
    print(f"Gear Train function value at [0.5]*4: {values[0]}")

Output:

.. code-block:: console

    Gear Train function value at [0.5]*4: tensor([-0.7323])