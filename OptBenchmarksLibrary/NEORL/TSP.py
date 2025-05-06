from ..base import BenchmarkProblem
import torch
from typing import Tuple
# from neorl.benchmarks import TSP

class TSP_51Cities(BenchmarkProblem):

    r'''
    Travelling Salesman Problem (TSP) with 51 cities.
    https://neorl.readthedocs.io/en/latest/examples/ex1.html#problem-description
    '''

    tags = {"single_objective", "unconstrained", "discrete", "TSP"}

    available_dimensions = [51, 100]
    num_objectives = 1

    def __init__(self):
        self.city_loc_list = [
            [37, 52], [49, 49], [52, 64], [20, 26], [40, 30], [21, 47], [17, 63], [31, 62],
            [52, 33], [51, 21], [42, 41], [31, 32], [5, 25], [12, 42], [36, 16], [52, 41],
            [27, 23], [17, 33], [13, 13], [57, 58], [62, 42], [42, 57], [16, 57], [8 ,52],
            [7 ,38], [27, 68], [30, 48], [43, 67], [58, 48], [58, 27], [37, 69], [38, 46],
            [46, 10],[61,33],[62,63],[63,69],[32,22],[45,35],[59,15],[5 ,6],[10 ,17],[21 ,10],
            [5 ,64],[30 ,15],[39 ,10],[32 ,39],[25 ,32],[25 ,55],[48 ,28],[56 ,37],[30 ,40]
        ]
        optimum = [1,22,8,26,31,28,3,36,35,20,2,29,21,16,50,34,30,9,49,10,39,33,45,15,44,42,40,19,41,13,25,14,24,43,7,23,48
                             ,6,27,51,46,12,47,18,4,17,37,5,38,11,32]
        super().__init__(dim = 51, 
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [list(range(1, 51))]*51, 
                         x_opt=[optimum])

    def _evaluate_implementation(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the total distance of a TSP tour.
        
        :param X: (torch.Tensor) a tensor of city indices representing the tour order
        :return: total distance traveled (torch.Tensor)
        """
        env = TSP(city_loc_list=self.city_loc_list, optimum_tour_city=[], episode_length=2)

        return env.evaluate(X)