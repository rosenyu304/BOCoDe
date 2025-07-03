import torch

from ..base import BenchmarkProblem, DataType


class Truss120D(BenchmarkProblem):
    """
    Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
    "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms
    for constrained design optimization of discrete truss structures,"
    Computers & Structures, Volume 212.
    """

    # 120D objective, 121 constraints, X = n-by-120

    available_dimensions = 120
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 121

    tags = {"single_objective", "constrained", "120D", "extra_imports"}

    def __init__(self):
        super().__init__(
            dim=120, num_objectives=1, num_constraints=121, bounds=[(0.775, 20)] * 120
        )

    def _evaluate_implementation(self, X, version="4_forces", to_verify=True):
        X = super().scale(X, to_verify)

        # import slientruss3d
        from slientruss3d.truss import Truss
        from slientruss3d.type import MemberType, SupportType

        def Truss120bar(A, E, Rho):
            # -------------------- Global variables --------------------
            # TEST_OUTPUT_FILE    = f"./test_output.json"
            TRUSS_DIMENSION = 3
            # ----------------------------------------------------------

            # Truss object:
            truss = Truss(dim=TRUSS_DIMENSION)

            # Truss settings:
            joints = [
                (0.0, 0.0, 275.59),
                (273.26, 0.0, 230.31),
                (236.65010183813573, 136.62999999999997, 230.31),
                (136.62999999999997, 236.65010183813573, 230.31),
                (0.0, 273.26, 230.31),
                (-136.62999999999997, 236.65010183813573, 230.31),
                (-236.65010183813573, 136.62999999999997, 230.31),
                (-273.26, 0.0, 230.31),
                (-236.65010183813573, -136.62999999999997, 230.31),
                (-136.62999999999997, -236.65010183813573, 230.31),
                (0.0, -273.26, 230.31),
                (136.62999999999997, -236.65010183813573, 230.31),
                (236.65010183813573, -136.62999999999997, 230.31),
                (492.12, 0.0, 118.11),
                (475.3514176333763, 127.37002847585251, 118.11),
                (426.18842171039796, 246.05999999999997, 118.11),
                (347.9813891575237, 347.9813891575237, 118.11),
                (246.05999999999997, 426.18842171039796, 118.11),
                (127.37002847585251, 475.3514176333763, 118.11),
                (0.0, 492.12, 118.11),
                (-127.37002847585251, 475.3514176333763, 118.11),
                (-246.05999999999997, 426.18842171039796, 118.11),
                (-347.9813891575237, 347.9813891575237, 118.11),
                (-426.18842171039796, 246.05999999999997, 118.11),
                (-475.3514176333763, 127.37002847585251, 118.11),
                (-492.12, 0.0, 118.11),
                (-475.3514176333763, -127.37002847585251, 118.11),
                (-426.18842171039796, -246.05999999999997, 118.11),
                (-347.9813891575237, -347.9813891575237, 118.11),
                (-246.05999999999997, -426.18842171039796, 118.11),
                (-127.37002847585251, -475.3514176333763, 118.11),
                (0.0, -492.12, 118.11),
                (127.37002847585251, -475.3514176333763, 118.11),
                (246.05999999999997, -426.18842171039796, 118.11),
                (347.9813891575237, -347.9813891575237, 118.11),
                (426.18842171039796, -246.05999999999997, 118.11),
                (475.3514176333763, -127.37002847585251, 118.11),
                (625.59, 0.0, 0.0),
                (541.7768323535071, 312.79499999999996, 0.0),
                (312.79499999999996, 541.7768323535071, 0.0),
                (0.0, 625.59, 0.0),
                (-312.79499999999996, 541.7768323535071, 0.0),
                (-541.7768323535071, 312.79499999999996, 0.0),
                (-625.59, 0.0, 0.0),
                (-541.7768323535071, -312.79499999999996, 0.0),
                (-312.79499999999996, -541.7768323535071, 0.0),
                (0.0, -625.59, 0.0),
                (312.79499999999996, -541.7768323535071, 0.0),
                (541.7768323535071, -312.79499999999996, 0.0),
            ]

            supports = [
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.NO,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
            ]

            # print(len(joints))
            # print(len(supports))

            forces = [
                (0, (0.0, 0.0, -13490.0)),
                (1, (0.0, 0.0, -6744.0)),
                (2, (0.0, 0.0, -6744.0)),
                (3, (0.0, 0.0, -6744.0)),
                (4, (0.0, 0.0, -6744.0)),
                (5, (0.0, 0.0, -6744.0)),
                (6, (0.0, 0.0, -6744.0)),
                (7, (0.0, 0.0, -6744.0)),
                (8, (0.0, 0.0, -6744.0)),
                (9, (0.0, 0.0, -6744.0)),
                (10, (0.0, 0.0, -6744.0)),
                (11, (0.0, 0.0, -6744.0)),
                (12, (0.0, 0.0, -6744.0)),
                (13, (0.0, 0.0, -6744.0)),
                (14, (0.0, 0.0, -2248.0)),
                (15, (0.0, 0.0, -2248.0)),
                (16, (0.0, 0.0, -2248.0)),
                (17, (0.0, 0.0, -2248.0)),
                (18, (0.0, 0.0, -2248.0)),
                (19, (0.0, 0.0, -2248.0)),
                (20, (0.0, 0.0, -2248.0)),
                (21, (0.0, 0.0, -2248.0)),
                (22, (0.0, 0.0, -2248.0)),
                (23, (0.0, 0.0, -2248.0)),
                (24, (0.0, 0.0, -2248.0)),
                (25, (0.0, 0.0, -2248.0)),
                (26, (0.0, 0.0, -2248.0)),
                (27, (0.0, 0.0, -2248.0)),
                (28, (0.0, 0.0, -2248.0)),
                (29, (0.0, 0.0, -2248.0)),
                (30, (0.0, 0.0, -2248.0)),
                (31, (0.0, 0.0, -2248.0)),
                (32, (0.0, 0.0, -2248.0)),
                (33, (0.0, 0.0, -2248.0)),
                (34, (0.0, 0.0, -2248.0)),
                (35, (0.0, 0.0, -2248.0)),
                (36, (0.0, 0.0, -2248.0)),
            ]

            members = [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (0, 6),
                (0, 7),
                (0, 8),
                (0, 9),
                (0, 10),
                (0, 11),
                (0, 12),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 1),
                (1, 13),
                (2, 15),
                (3, 17),
                (4, 19),
                (5, 21),
                (6, 23),
                (7, 25),
                (8, 27),
                (9, 29),
                (10, 31),
                (11, 33),
                (12, 35),
                (1, 14),
                (2, 14),
                (2, 16),
                (3, 16),
                (3, 18),
                (4, 18),
                (4, 20),
                (5, 20),
                (5, 22),
                (6, 22),
                (6, 24),
                (7, 24),
                (7, 26),
                (8, 26),
                (8, 28),
                (9, 28),
                (9, 30),
                (10, 30),
                (10, 32),
                (11, 32),
                (11, 34),
                (12, 34),
                (12, 36),
                (1, 36),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (26, 27),
                (27, 28),
                (28, 29),
                (29, 30),
                (30, 31),
                (31, 32),
                (32, 33),
                (33, 34),
                (34, 35),
                (35, 36),
                (36, 13),
                (13, 37),
                (15, 38),
                (17, 39),
                (19, 40),
                (21, 41),
                (23, 42),
                (25, 43),
                (27, 44),
                (29, 45),
                (31, 46),
                (33, 47),
                (35, 48),
                (14, 37),
                (14, 38),
                (16, 38),
                (16, 39),
                (18, 39),
                (18, 40),
                (20, 40),
                (20, 41),
                (22, 41),
                (22, 42),
                (24, 42),
                (24, 43),
                (26, 43),
                (26, 44),
                (28, 44),
                (28, 45),
                (30, 45),
                (30, 46),
                (32, 46),
                (32, 47),
                (34, 47),
                (34, 48),
                (36, 48),
                (36, 37),
            ]

            # memberType : Member type which contain the information about
            # 1) cross-sectional area,
            # 2) Young's modulus,
            # 3) density of this member.

            # Read data in this [.py]:
            for joint, support in zip(joints, supports):
                truss.AddNewJoint(joint, support)

            for jointID, force in forces:
                truss.AddExternalForce(jointID, force)

            index = 0
            for jointID0, jointID1 in members:
                # memberType = MemberType(A[index].item(), 30450000, 0.288)
                # print(A.shape)
                memberType = MemberType(A[index].item(), 30450000, 0.288)

                if (E is not None) and (Rho is not None):
                    memberType = MemberType(
                        A[index].item(), E[index].item(), Rho[index].item()
                    )
                elif (E is not None) and (Rho is None):
                    memberType = MemberType(A[index].item(), E[index].item(), 0.288)
                elif (E is None) and (Rho is not None):
                    memberType = MemberType(
                        A[index].item(), 30450000, Rho[index].item()
                    )

                truss.AddNewMember(jointID0, jointID1, memberType)
                index += 1

            # Do direct stiffness method:
            truss.Solve()

            # Dump all the structural analysis results into a .json file:
            # truss.DumpIntoJSON(TEST_OUTPUT_FILE)

            # Get result of structural analysis:
            displace, forces, stress, resistance = (
                truss.GetDisplacements(),
                truss.GetInternalForces(),
                truss.GetInternalStresses(),
                truss.GetResistances(),
            )
            return displace, forces, stress, resistance, truss, truss.weight

        E = None
        Rho = None

        n = X.size(0)

        fx = torch.zeros(n, 1)

        # 120 bar stress constraints, 1 displacement constraints
        gx = torch.zeros(n, 121)

        for ii in range(n):
            # print(ii)
            # print(A[ii,:].shape)
            displace, _, stress, _, _, weights = Truss120bar(X[ii, :], None, None)

            if (E is not None) and (Rho is not None):
                displace, _, stress, _, _, weights = Truss120bar(
                    X[ii, :], E[ii, :], Rho[ii, :]
                )
            elif (E is not None) and (Rho is None):
                displace, _, stress, _, _, weights = Truss120bar(
                    X[ii, :], E[ii, :], None
                )
            elif (E is None) and (Rho is not None):
                displace, _, stress, _, _, weights = Truss120bar(
                    X[ii, :], None, Rho[ii, :]
                )

            fx[ii, 0] = -weights  # Negate for maximizing optimization

            # Max stress less than 34800
            for ss in range(120):
                gx[ii, ss] = abs(stress[ss]) - 34800

            # Max displacement in x and y direction less than
            MAX_DIST = 0
            for dd in range(len(displace)):
                if max(displace[dd]) > MAX_DIST:
                    MAX_DIST = max(abs(displace[dd]))
            gx[ii, 120] = MAX_DIST - 0.1969

        return gx, fx
