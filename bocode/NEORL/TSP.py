from typing import Tuple

import numpy as np
import torch

from ..base import BenchmarkProblem, DataType
from .neorl_lib.tsp import TSP


class TSP_51Cities(BenchmarkProblem):
    """
    Travelling Salesman Problem (TSP) with 51 cities.
    https://neorl.readthedocs.io/en/latest/examples/ex1.html#problem-description
    """

    tags = {"single_objective", "unconstrained", "discrete", "TSP"}

    available_dimensions = 51
    input_type = DataType.DISCRETE
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        self.city_loc_list = [
            [37, 52],
            [49, 49],
            [52, 64],
            [20, 26],
            [40, 30],
            [21, 47],
            [17, 63],
            [31, 62],
            [52, 33],
            [51, 21],
            [42, 41],
            [31, 32],
            [5, 25],
            [12, 42],
            [36, 16],
            [52, 41],
            [27, 23],
            [17, 33],
            [13, 13],
            [57, 58],
            [62, 42],
            [42, 57],
            [16, 57],
            [8, 52],
            [7, 38],
            [27, 68],
            [30, 48],
            [43, 67],
            [58, 48],
            [58, 27],
            [37, 69],
            [38, 46],
            [46, 10],
            [61, 33],
            [62, 63],
            [63, 69],
            [32, 22],
            [45, 35],
            [59, 15],
            [5, 6],
            [10, 17],
            [21, 10],
            [5, 64],
            [30, 15],
            [39, 10],
            [32, 39],
            [25, 32],
            [25, 55],
            [48, 28],
            [56, 37],
            [30, 40],
        ]

        optimum = [
            1,
            22,
            8,
            26,
            31,
            28,
            3,
            36,
            35,
            20,
            2,
            29,
            21,
            16,
            50,
            34,
            30,
            9,
            49,
            10,
            39,
            33,
            45,
            15,
            44,
            42,
            40,
            19,
            41,
            13,
            25,
            14,
            24,
            43,
            7,
            23,
            48,
            6,
            27,
            51,
            46,
            12,
            47,
            18,
            4,
            17,
            37,
            5,
            38,
            11,
            32,
        ]
        self.optimum_cities = optimum

        super().__init__(
            dim=51,
            num_objectives=1,
            num_constraints=0,
            bounds=[list(range(1, 51))] * 51,
            x_opt=[optimum],
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the total distance of a TSP tour.

        :param X: (torch.Tensor) a tensor of city indices representing the tour order
        :return: total distance traveled (torch.Tensor)
        """

        env = TSP(
            city_loc_list=self.city_loc_list,
            optimum_tour_city=None,
            episode_length=2,
            method="ppo",
        )

        fx = torch.zeros(X.shape[0], dtype=torch.float32)
        for i in range(X.shape[0]):
            tour = X[i].tolist()
            tour = [int(x) for x in tour]
            tour_map = np.full((len(tour), 4), -1e6, dtype=np.float32)
            for j, city_idx in enumerate(tour):
                coords = env.city_library[str(city_idx)]
                tour_map[j][0] = coords[0]
                tour_map[j][1] = coords[1]
            fx[i] = env.Compute_tour_cost(tour=tour_map)

        return None, fx.unsqueeze(-1)


class TSP_100Cities(BenchmarkProblem):
    """
    Travelling Salesman Problem (TSP) with 100 cities.
    https://neorl.readthedocs.io/en/latest/examples/ex1.html#problem-description
    """

    tags = {"single_objective", "unconstrained", "discrete", "TSP"}

    available_dimensions = 100
    input_type = DataType.DISCRETE
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        # 100 cities
        self.city_loc_list = [
            [-47, 2],
            [49, -21],
            [35, -47],
            [30, -47],
            [-39, -50],
            [-35, -27],
            [-34, 9],
            [-11, -8],
            [32, -44],
            [1, 35],
            [36, 37],
            [12, 37],
            [37, 36],
            [-26, -8],
            [-21, 32],
            [-29, 13],
            [26, -50],
            [-7, -36],
            [-34, -2],
            [21, -40],
            [-25, 46],
            [-17, 8],
            [21, 27],
            [-31, -14],
            [-15, -44],
            [-33, -34],
            [-49, 45],
            [-40, -1],
            [-40, -33],
            [-39, -26],
            [-17, -16],
            [17, -20],
            [4, -11],
            [22, 34],
            [28, 24],
            [-39, 37],
            [25, 4],
            [-35, 14],
            [34, -5],
            [49, -43],
            [34, -29],
            [-4, -50],
            [0, -14],
            [48, -25],
            [-50, -5],
            [-26, 0],
            [-13, 21],
            [-6, -41],
            [40, -33],
            [12, -48],
            [-38, 16],
            [-26, -38],
            [-42, 16],
            [13, 8],
            [4, -8],
            [-46, -20],
            [-25, 36],
            [22, 21],
            [43, -5],
            [-24, 0],
            [-12, -32],
            [47, 49],
            [31, -35],
            [42, 13],
            [-45, -45],
            [-48, -14],
            [28, 23],
            [23, -43],
            [30, -25],
            [25, 34],
            [-7, 32],
            [-48, 42],
            [1, -26],
            [-45, 32],
            [-20, 35],
            [-12, 21],
            [-41, -49],
            [-35, 32],
            [-43, 44],
            [-43, 47],
            [27, 20],
            [-8, -9],
            [37, -11],
            [-18, 16],
            [-41, 43],
            [-30, 29],
            [-31, -19],
            [48, 22],
            [-45, -19],
            [-15, 30],
            [10, -8],
            [40, -33],
            [20, 20],
            [-22, 33],
            [42, -37],
            [0, -8],
            [-50, 11],
            [37, -27],
            [39, -43],
            [-7, 32],
        ]

        optimum = [
            1,
            97,
            53,
            51,
            38,
            16,
            7,
            28,
            19,
            46,
            60,
            22,
            84,
            76,
            47,
            86,
            78,
            36,
            74,
            72,
            27,
            80,
            79,
            85,
            21,
            57,
            94,
            15,
            75,
            90,
            71,
            100,
            10,
            12,
            34,
            70,
            11,
            13,
            62,
            88,
            64,
            81,
            67,
            35,
            23,
            58,
            93,
            54,
            37,
            39,
            83,
            59,
            2,
            44,
            98,
            41,
            69,
            63,
            49,
            92,
            95,
            40,
            99,
            3,
            9,
            4,
            17,
            68,
            20,
            50,
            42,
            25,
            48,
            18,
            61,
            73,
            32,
            91,
            55,
            33,
            43,
            96,
            82,
            8,
            31,
            14,
            24,
            87,
            6,
            26,
            52,
            5,
            77,
            65,
            29,
            30,
            89,
            56,
            66,
            45,
        ]
        self.optimum_cities = optimum

        super().__init__(
            dim=100,
            num_objectives=1,
            num_constraints=0,
            bounds=[list(range(1, 100))] * 100,
            x_opt=[optimum],
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the total distance of a TSP tour.

        :param X: (torch.Tensor) a tensor of city indices representing the tour order
        :return: total distance traveled (torch.Tensor)
        """

        env = TSP(
            city_loc_list=self.city_loc_list,
            optimum_tour_city=None,
            episode_length=2,
            method="ppo",
        )

        fx = torch.zeros(X.shape[0], dtype=torch.float32)
        for i in range(X.shape[0]):
            tour = X[i].tolist()
            tour = [int(x) for x in tour]
            tour_map = np.full((len(tour), 4), -1e6, dtype=np.float32)
            for j, city_idx in enumerate(tour):
                coords = env.city_library[str(city_idx)]
                tour_map[j][0] = coords[0]
                tour_map[j][1] = coords[1]
            fx[i] = env.Compute_tour_cost(tour=tour_map)

        return None, fx.unsqueeze(-1)
