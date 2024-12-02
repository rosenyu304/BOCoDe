import Ackley
import Bukin
import CantileverBeam
import Car
import CompressionSpring
import DixonPrice
import EulerBernoulliBeamBending
import GearTrain
import GKXWC1
import GKXWC2
import Goldstein
import Griewank
import HeatExchanger
import JLH1
import JLH2
import KeaneBump
import Levy
# import Mazda
# import Mazda_softpen
import Michalewicz
import MOPTA08Car
import MOPTA08Car_softpen
import PressureVessel
import ReinforcedConcreteBeam
import Rosenbrock
import SpeedReducer
# import StyblinskiTang_continuous
# import StyblinskiTang_mixed
import ThreeTruss
import Truss10D
import Truss25D
import Truss72D
import Truss120D
import Truss200D
import WeldedBeam


problem_database = {Ackley.Ackley: Ackley.Ackley.tags,
                    Bukin.Bukin: Bukin.Bukin.tags,
                    CantileverBeam.CantileverBeam: CantileverBeam.CantileverBeam.tags,
                    Car.Car: Car.Car.tags,
                    CompressionSpring.CompressionSpring: CompressionSpring.CompressionSpring.tags,
                    DixonPrice.DixonPrice: DixonPrice.DixonPrice.tags,
                    EulerBernoulliBeamBending.EulerBernoulliBeamBending: EulerBernoulliBeamBending.EulerBernoulliBeamBending.tags,
                    GearTrain.GearTrain: GearTrain.GearTrain.tags,
                    GKXWC1.GKXWC1: GKXWC1.GKXWC1.tags,
                    GKXWC2.GKXWC2: GKXWC2.GKXWC2.tags,
                    Goldstein.Goldstein: Goldstein.Goldstein.tags,
                    Griewank.Griewank: Griewank.Griewank.tags,
                    HeatExchanger.HeatExchanger: HeatExchanger.HeatExchanger.tags,
                    JLH1.JLH1: JLH1.JLH1.tags,
                    JLH2.JLH2: JLH2.JLH2.tags,
                    KeaneBump.KeaneBump: KeaneBump.KeaneBump.tags,
                    Levy.Levy: Levy.Levy.tags,
                    # Mazda.Mazda: Mazda.Mazdatags,
                    # Mazda_softpen.Mazda_softpen: Mazda_softpen.Mazda_softpen.tags,
                    Michalewicz.Michalewicz: Michalewicz.Michalewicz.tags,
                    MOPTA08Car.MOPTA08Car: MOPTA08Car.MOPTA08Car.tags,
                    MOPTA08Car_softpen.MOPTA08Car_softpen: MOPTA08Car_softpen.MOPTA08Car_softpen.tags,
                    PressureVessel.PressureVessel: PressureVessel.PressureVessel.tags,
                    ReinforcedConcreteBeam.ReinforcedConcreteBeam: ReinforcedConcreteBeam.ReinforcedConcreteBeam.tags,
                    Rosenbrock.Rosenbrock: Rosenbrock.Rosenbrock.tags,
                    SpeedReducer.SpeedReducer: SpeedReducer.SpeedReducer.tags,
                    # StyblinskiTang_continuous.StyblinskiTang_continuous: StyblinskiTang_continuous.StyblinskiTang_continuous.tags,
                    # StyblinskiTang_mixed.StyblinskiTang_mixed: StyblinskiTang_mixed.StyblinskiTang_mixed.tags,
                    ThreeTruss.ThreeTruss: ThreeTruss.ThreeTruss.tags,
                    Truss10D.Truss10D: Truss10D.Truss10D.tags,
                    Truss25D.Truss25D: Truss25D.Truss25D.tags,
                    Truss72D.Truss72D: Truss72D.Truss72D.tags,
                    Truss120D.Truss120D: Truss120D.Truss120D.tags,
                    Truss200D.Truss200D: Truss200D.Truss200D.tags,
                    WeldedBeam.WeldedBeam: WeldedBeam.WeldedBeam.tags}

def find_benchmark_problems(tags = None, extra_imports = False):
    """
    Returns a set of Benchmark Problems that each have all of the tags from
    at least one of tag's inner lists. (The inner list acts as an AND,
    and the list of lists acts as an OR.)

    Parameters:
        tags (2D list): a list of specified tags
            - tag options: "single_objective", "constrained", "unconstrained",
              "continuous", "mixed", "xD" (x = positive int or N for any)

    Returns:
        return_probs (list): satisfactory Benchmark Problems
    """
    return_probs = set()

    for set_of_tags in tags:
        for prob in problem_database:
            if not (extra_imports == False and "extra_imports" in problem_database[prob]):
                to_add = True
                for tag in set_of_tags:
                    if to_add and tag not in problem_database[prob]:
                        to_add = False
                if to_add:
                    return_probs.add(prob)

    return list(return_probs)


# l = find_benchmark_problems([["ND"]], extra_imports = True)
# print(l)

# a = l[2](dim=6)
# X = ([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]])
# gx, fx = a.evaluate(X, to_verify = True)
# print(gx, fx)
