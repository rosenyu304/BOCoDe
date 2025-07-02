.. _engineering_benchmarks:

Engineering Benchmarks
=================

The Engineering benchmark collection contains various Engineering-related functions.

Available Problems
----------------


* :code:`bocode.Engineering.CarSideImpact`
* :code:`bocode.Engineering.EulerBernoulliBeamBending`
* :code:`bocode.Engineering.GearTrain`
* :code:`bocode.Engineering.Mazda_SCA`
* :code:`bocode.Engineering.Mazda`
* :code:`bocode.Engineering.MOPTA08Car`
* :code:`bocode.Engineering.RobotPush`
* :code:`bocode.Engineering.Rover`
* :code:`bocode.Engineering.Truss10D`
* :code:`bocode.Engineering.Truss25D`
* :code:`bocode.Engineering.TwoBarTruss`
* :code:`bocode.Engineering.WaterProblem`
* :code:`bocode.Engineering.WaterResources`
* Bayesian CHT Functions (:code:`bocode.Engineering.BayesianCHT`) 
    `Source <https://link.springer.com/article/10.1007/s00158-024-03859-y>`_

     Y.-K. Tsai and R. J. Malak Jr, “Surrogate-assisted constraint-handling technique for constrained parametric multi-objective optimization,” Structural and Multidisciplinary Optimization, 2024. 
    
    * :code:`bocode.Engineering.BayesianCHT.NonLinearConstraintProblemA3`
    * :code:`bocode.Engineering.BayesianCHT.NonLinearConstraintProblemA4`
    * :code:`bocode.Engineering.BayesianCHT.NonLinearConstraintProblemA7`
    * :code:`bocode.Engineering.BayesianCHT.NonLinearConstraintProblemA8`
    * :code:`bocode.Engineering.BayesianCHT.NonLinearConstraintProblemB3`
    * :code:`bocode.Engineering.BayesianCHT.NonLinearConstraintProblemB4`
    * :code:`bocode.Engineering.BayesianCHT.NonLinearConstraintProblemB7`
    * :code:`bocode.Engineering.BayesianCHT.NonLinearConstraintProblemB8`
* Mujoco Functions (:code:`bocode.Engineering.Gym`) 
    `Source <https://gymnasium.farama.org/environments/mujoco/>`_

     E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., pp. 5026–5033, 2012. doi: 10.1109/IROS.2012.6386109.

    * :code:`bocode.Engineering.Gym.AntProblem`
    * :code:`bocode.Engineering.Gym.HalfCheetahProblem`
    * :code:`bocode.Engineering.Gym.HopperProblem`
    * :code:`bocode.Engineering.Gym.HumanoidProblem`
    * :code:`bocode.Engineering.Gym.HumanoidStandupProblem`
    * :code:`bocode.Engineering.Gym.InvertedDoublePendulumProblem`
    * :code:`bocode.Engineering.Gym.InvertedPendulumProblem`
    * :code:`bocode.Engineering.Gym.PusherProblem`
    * :code:`bocode.Engineering.Gym.ReacherProblem`
    * :code:`bocode.Engineering.Gym.SwimmerProblem`
    * :code:`bocode.Engineering.Gym.Walker2DProblem`
    * :code:`bocode.Engineering.Gym.SwimmerPolicySearchProblem`
    * :code:`bocode.Engineering.Gym.AntPolicySearchProblem`
    * :code:`bocode.Engineering.Gym.HalfCheetahPolicySearchProblem`
    * :code:`bocode.Engineering.Gym.HopperPolicySearchProblem`
    * :code:`bocode.Engineering.Gym.Walker2DPolicySearchProblem`

Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Create a Botorch benchmark problem
    problem = bocode.Engineering.GearTrain()
    
    # Get problem information
    bounds = problem.bounds
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    values, constraints = problem.evaluate(x)
    
    print(f"Gear Train function value at [0.5]*4: {values[0]}")

Output:

.. code-block:: console

    Gear Train function value at [0.5]*4: tensor([-0.7323])