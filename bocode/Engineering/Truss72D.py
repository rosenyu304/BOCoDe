import torch

from ..base import BenchmarkProblem, DataType


class Truss72D_FourForces(BenchmarkProblem):
    """
    Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
    "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms
    for constrained design optimization of discrete truss structures,"
    Computers & Structures, Volume 212.
    """

    # 72D objective, 88 constraints, X = n-by-72

    tags = {"single_objective", "constrained", "72D", "extra_imports"}

    available_dimensions = 72
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 88

    def __init__(self):
        super().__init__(
            dim=72, num_objectives=1, num_constraints=88, bounds=[(0.1, 33.5)] * 72
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        # import slientruss3d
        from slientruss3d.truss import Truss
        from slientruss3d.type import MemberType, SupportType

        def Truss72bar(A, E, Rho, version="4_forces"):
            # -------------------- Global variables --------------------
            # TEST_OUTPUT_FILE    = f"./test_output.json"
            TRUSS_DIMENSION = 3
            # ----------------------------------------------------------

            # Truss object:
            truss = Truss(dim=TRUSS_DIMENSION)

            # Truss settings:
            joints = [
                (0.0, 0.0, 0.0),
                (120.0, 0.0, 0.0),
                (120.0, 120.0, 0.0),
                (0.0, 120.0, 0.0),
                (0.0, 0.0, 60.0),
                (120.0, 0.0, 60.0),
                (120.0, 120.0, 60.0),
                (0.0, 120.0, 60.0),
                (0.0, 0.0, 120.0),
                (120.0, 0.0, 120.0),
                (120.0, 120.0, 120.0),
                (0.0, 120.0, 120.0),
                (0.0, 0.0, 180.0),
                (120.0, 0.0, 180.0),
                (120.0, 120.0, 180.0),
                (0.0, 120.0, 180.0),
                (0.0, 0.0, 240.0),
                (120.0, 0.0, 240.0),
                (120.0, 120.0, 240.0),
                (0.0, 120.0, 240.0),
            ]

            supports = supports = [
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
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
            ]

            forces = [
                (0, (0.0, 0.0, 0.0)),
                (1, (0.0, 0.0, 0.0)),
                (2, (0.0, 0.0, 0.0)),
                (3, (0.0, 0.0, 0.0)),
                (4, (0.0, 0.0, 0.0)),
                (5, (0.0, 0.0, 0.0)),
                (6, (0.0, 0.0, 0.0)),
                (7, (0.0, 0.0, 0.0)),
                (8, (0.0, 0.0, 0.0)),
                (9, (0.0, 0.0, 0.0)),
                (10, (0.0, 0.0, 0.0)),
                (11, (0.0, 0.0, 0.0)),
                (12, (0.0, 0.0, 0.0)),
                (13, (0.0, 0.0, 0.0)),
                (14, (0.0, 0.0, 0.0)),
                (15, (0.0, 0.0, 0.0)),
                (16, (0.0, 0.0, -5000.0)),
                (17, (0.0, 0.0, -5000.0)),
                (18, (0.0, 0.0, -5000.0)),
                (19, (0.0, 0.0, -5000.0)),
            ]

            members = [
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
                (1, 4),
                (0, 5),
                (1, 6),
                (2, 5),
                (2, 7),
                (3, 6),
                (0, 7),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (4, 6),
                (5, 7),
                (4, 8),
                (5, 9),
                (6, 10),
                (7, 11),
                (5, 8),
                (4, 9),
                (5, 10),
                (6, 9),
                (6, 11),
                (7, 10),
                (4, 11),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 8),
                (8, 10),
                (9, 11),
                (8, 12),
                (9, 13),
                (10, 14),
                (11, 15),
                (9, 12),
                (8, 13),
                (9, 14),
                (10, 13),
                (10, 15),
                (11, 14),
                (8, 15),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 12),
                (12, 14),
                (13, 15),
                (12, 16),
                (13, 17),
                (14, 18),
                (15, 19),
                (13, 16),
                (12, 17),
                (13, 18),
                (14, 17),
                (14, 19),
                (15, 18),
                (12, 19),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 16),
                (16, 18),
                (17, 19),
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
                # memberType = MemberType(A[index].item(), 10000000.0, 0.1)

                memberType = MemberType(A[index].item(), 10000000.0, 0.1)

                if (E is not None) and (Rho is not None):
                    memberType = MemberType(
                        A[index].item(), E[index].item(), Rho[index].item()
                    )
                elif (E is not None) and (Rho is None):
                    memberType = MemberType(A[index].item(), E[index].item(), 0.1)
                elif (E is None) and (Rho is not None):
                    memberType = MemberType(
                        A[index].item(), 10000000.0, Rho[index].item()
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

        if X.size(1) == 72:
            A = X
        elif X.size(1) == 16:
            # Bars in 16 groups because of symmetry
            # (1) A1–A4, (2) A5–A12, (3) A13–A16, (4) A17–A18, (5) A19–A22, (6) A23–A30, (7) A31–A34, (8) A35–A36,
            # (9) A37–A40, (10) A41–A48, (11) A49–A52, (12) A53–A54, (13) A55–A58, (14) A59–A66 (15), A67–A70, and (16) A71–A72.
            A = torch.zeros(X.size(0), 72)
            A[:, 0:4] = X[:, 0]
            A[:, 4:12] = X[:, 1]
            A[:, 12:16] = X[:, 2]
            A[:, 16:18] = X[:, 3]
            A[:, 18:22] = X[:, 4]
            A[:, 22:30] = X[:, 5]
            A[:, 30:34] = X[:, 6]
            A[:, 34:36] = X[:, 7]
            A[:, 36:40] = X[:, 8]
            A[:, 40:48] = X[:, 9]
            A[:, 48:52] = X[:, 10]
            A[:, 52:54] = X[:, 11]
            A[:, 54:58] = X[:, 12]
            A[:, 58:66] = X[:, 13]
            A[:, 66:70] = X[:, 14]
            A[:, 70:72] = X[:, 15]

        E = 1e7 * torch.ones(72)
        Rho = 0.1 * torch.ones(72)

        n = A.size(0)

        fx = torch.zeros(n, 1)

        # 72 bar stress constraints, 16 displacement constraints
        gx = torch.zeros(n, 88)

        for ii in range(n):
            displace, _, stress, _, _, weights = Truss72bar(A[ii, :], E, Rho)

            fx[ii, 0] = -weights  # Negate for maximizing optimization

            # Max stress less than 25000
            for ss in range(72):
                gx[ii, ss] = abs(stress[ss]) - 25000

            # Max displacement in x and y direction less than .25 inches
            for dd in range(4, 20):  # 16 free nodes
                gx[ii, 72 + dd - 4] = (
                    max(abs(displace[dd][0]), abs(displace[dd][1])) - 0.25
                )

            return gx, fx


class Truss72D_SingleForce(BenchmarkProblem):
    """
    Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
    "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms
    for constrained design optimization of discrete truss structures,"
    Computers & Structures, Volume 212.
    """

    # 72D objective, 88 constraints, X = n-by-72

    tags = {"single_objective", "constrained", "72D", "extra_imports"}

    available_dimensions = 72
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 88

    def __init__(self):
        super().__init__(
            dim=72, num_objectives=1, num_constraints=88, bounds=[(0.1, 33.5)] * 72
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        # import slientruss3d
        from slientruss3d.truss import Truss
        from slientruss3d.type import MemberType, SupportType

        def Truss72bar(A, E, Rho):
            # -------------------- Global variables --------------------
            # TEST_OUTPUT_FILE    = f"./test_output.json"
            TRUSS_DIMENSION = 3
            # ----------------------------------------------------------

            # Truss object:
            truss = Truss(dim=TRUSS_DIMENSION)

            # Truss settings:
            joints = [
                (0.0, 0.0, 0.0),
                (120.0, 0.0, 0.0),
                (120.0, 120.0, 0.0),
                (0.0, 120.0, 0.0),
                (0.0, 0.0, 60.0),
                (120.0, 0.0, 60.0),
                (120.0, 120.0, 60.0),
                (0.0, 120.0, 60.0),
                (0.0, 0.0, 120.0),
                (120.0, 0.0, 120.0),
                (120.0, 120.0, 120.0),
                (0.0, 120.0, 120.0),
                (0.0, 0.0, 180.0),
                (120.0, 0.0, 180.0),
                (120.0, 120.0, 180.0),
                (0.0, 120.0, 180.0),
                (0.0, 0.0, 240.0),
                (120.0, 0.0, 240.0),
                (120.0, 120.0, 240.0),
                (0.0, 120.0, 240.0),
            ]

            supports = supports = [
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
                SupportType.PIN,
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
            ]

            forces = [
                (0, (0.0, 0.0, 0.0)),
                (1, (0.0, 0.0, 0.0)),
                (2, (0.0, 0.0, 0.0)),
                (3, (0.0, 0.0, 0.0)),
                (4, (0.0, 0.0, 0.0)),
                (5, (0.0, 0.0, 0.0)),
                (6, (0.0, 0.0, 0.0)),
                (7, (0.0, 0.0, 0.0)),
                (8, (0.0, 0.0, 0.0)),
                (9, (0.0, 0.0, 0.0)),
                (10, (0.0, 0.0, 0.0)),
                (11, (0.0, 0.0, 0.0)),
                (12, (0.0, 0.0, 0.0)),
                (13, (0.0, 0.0, 0.0)),
                (14, (0.0, 0.0, 0.0)),
                (15, (0.0, 0.0, 0.0)),
                (16, (5000.0, 5000.0, -5000.0)),
                (17, (0.0, 0.0, 0.0)),
                (18, (0.0, 0.0, 0.0)),
                (19, (0.0, 0.0, 0.0)),
            ]

            members = [
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
                (1, 4),
                (0, 5),
                (1, 6),
                (2, 5),
                (2, 7),
                (3, 6),
                (0, 7),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (4, 6),
                (5, 7),
                (4, 8),
                (5, 9),
                (6, 10),
                (7, 11),
                (5, 8),
                (4, 9),
                (5, 10),
                (6, 9),
                (6, 11),
                (7, 10),
                (4, 11),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 8),
                (8, 10),
                (9, 11),
                (8, 12),
                (9, 13),
                (10, 14),
                (11, 15),
                (9, 12),
                (8, 13),
                (9, 14),
                (10, 13),
                (10, 15),
                (11, 14),
                (8, 15),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 12),
                (12, 14),
                (13, 15),
                (12, 16),
                (13, 17),
                (14, 18),
                (15, 19),
                (13, 16),
                (12, 17),
                (13, 18),
                (14, 17),
                (14, 19),
                (15, 18),
                (12, 19),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 16),
                (16, 18),
                (17, 19),
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
                # memberType = MemberType(A[index].item(), 10000000.0, 0.1)

                memberType = MemberType(A[index].item(), 10000000.0, 0.1)

                if (E is not None) and (Rho is not None):
                    memberType = MemberType(
                        A[index].item(), E[index].item(), Rho[index].item()
                    )
                elif (E is not None) and (Rho is None):
                    memberType = MemberType(A[index].item(), E[index].item(), 0.1)
                elif (E is None) and (Rho is not None):
                    memberType = MemberType(
                        A[index].item(), 10000000.0, Rho[index].item()
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

        if X.size(1) == 72:
            A = X
        elif X.size(1) == 16:
            # Bars in 16 groups because of symmetry
            # (1) A1–A4, (2) A5–A12, (3) A13–A16, (4) A17–A18, (5) A19–A22, (6) A23–A30, (7) A31–A34, (8) A35–A36,
            # (9) A37–A40, (10) A41–A48, (11) A49–A52, (12) A53–A54, (13) A55–A58, (14) A59–A66 (15), A67–A70, and (16) A71–A72.
            A = torch.zeros(X.size(0), 72)
            A[:, 0:4] = X[:, 0]
            A[:, 4:12] = X[:, 1]
            A[:, 12:16] = X[:, 2]
            A[:, 16:18] = X[:, 3]
            A[:, 18:22] = X[:, 4]
            A[:, 22:30] = X[:, 5]
            A[:, 30:34] = X[:, 6]
            A[:, 34:36] = X[:, 7]
            A[:, 36:40] = X[:, 8]
            A[:, 40:48] = X[:, 9]
            A[:, 48:52] = X[:, 10]
            A[:, 52:54] = X[:, 11]
            A[:, 54:58] = X[:, 12]
            A[:, 58:66] = X[:, 13]
            A[:, 66:70] = X[:, 14]
            A[:, 70:72] = X[:, 15]

        E = 1e7 * torch.ones(72)
        Rho = 0.1 * torch.ones(72)

        n = A.size(0)

        fx = torch.zeros(n, 1)

        # 72 bar stress constraints, 16 displacement constraints
        gx = torch.zeros(n, 88)

        for ii in range(n):
            displace, _, stress, _, _, weights = Truss72bar(A[ii, :], E, Rho)

            fx[ii, 0] = -weights  # Negate for maximizing optimization

            # Max stress less than 25000
            for ss in range(72):
                gx[ii, ss] = abs(stress[ss]) - 25000

            # Max displacement in x and y direction less than .25 inches
            for dd in range(4, 20):  # 16 free nodes
                gx[ii, 72 + dd - 4] = (
                    max(abs(displace[dd][0]), abs(displace[dd][1])) - 0.25
                )

            return gx, fx
