import torch

from ..base import BenchmarkProblem, DataType


class Truss25D(BenchmarkProblem):
    """
    Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
    "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms
    for constrained design optimization of discrete truss structures,"
    Computers & Structures, Volume 212.
    """

    available_dimensions = 25
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 31

    def __init__(self):
        tags = [
            "Truss25D",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: Constrained (31)",
            "SPACE: Continuous",
            "SCALABLE: N/A",
            "IMPORTS: slientruss3d",
        ]

        super().__init__(
            dim=25,
            num_objectives=1,
            num_constraints=31,
            bounds=[(0.1, 3.4)] * 25,
            tags=tags,
        )

    @staticmethod
    def Truss25bar(A, E, Rho):
        # import slientruss3d
        from slientruss3d.truss import Truss
        from slientruss3d.type import MemberType, SupportType

        # -------------------- Global variables --------------------
        # TEST_OUTPUT_FILE    = f"./test_output.json"
        TRUSS_DIMENSION = 3
        # ----------------------------------------------------------

        # Truss object:
        truss = Truss(dim=TRUSS_DIMENSION)

        # Truss settings:
        joints = [
            (62.5, 100, 200),
            (137.5, 100, 200),
            (62.5, 137.5, 100),
            (137.5, 137.5, 100),
            (137.5, 62.5, 100),
            (62.5, 62.5, 100),
            (0, 200, 0),
            (200, 200, 0),
            (200, 0, 0),
            (0, 0, 0),
        ]
        supports = [
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
        ]
        forces = [
            (0, (1000, -10000, -10000)),
            (1, (0, -10000, -10000)),
            (2, (500, 0, 0)),
            (5, (600, 0, 0)),
        ]
        members = [
            (0, 1),
            (0, 3),
            (1, 2),
            (0, 4),
            (1, 5),
            (0, 2),
            (0, 5),
            (1, 3),
            (1, 4),
            (2, 5),
            (3, 4),
            (2, 3),
            (4, 5),
            (2, 9),
            (5, 6),
            (3, 8),
            (4, 7),
            (2, 7),
            (3, 6),
            (5, 8),
            (4, 9),
            (2, 6),
            (3, 7),
            (4, 8),
            (5, 9),
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
            # Default: 0.1, 1e7, .1

            memberType = MemberType(A[index].item(), 3e7, 0.283)

            if (E is not None) and (Rho is not None):
                memberType = MemberType(
                    A[index].item(), E[index].item(), Rho[index].item()
                )
            elif (E is not None) and (Rho is None):
                memberType = MemberType(A[index].item(), E[index].item(), 0.283)
            elif (E is None) and (Rho is not None):
                memberType = MemberType(A[index].item(), 3e7, Rho[index].item())

            # memberType = MemberType(A[index].item(), 1e7, .1)
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

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        if X.size(1) == 25:
            A = X
        elif X.size(1) == 8:
            # Bars in 8 groups because of symmetry
            # (1) A1, (2) A2–A5, (3) A6–A9, (4) A10–A11, (5) A12–A13, (6) A14–A17, (7) A18–A21 and (8) A22–A25.
            A = torch.zeros(X.size(0), 25)
            A[:, 0] = X[:, 0]
            A[:, 1:5] = X[:, 1]
            A[:, 5:9] = X[:, 2]
            A[:, 9:11] = X[:, 3]
            A[:, 11:13] = X[:, 4]
            A[:, 13:17] = X[:, 5]
            A[:, 17:21] = X[:, 6]
            A[:, 21:25] = X[:, 7]

        E = 1e7 * torch.ones(25)
        Rho = 0.1 * torch.ones(25)

        n = X.size(0)

        fx = torch.zeros(n, 1)

        # 25 bar stress constraints, 6 displacement constraints
        gx = torch.zeros(n, 31)

        for ii in range(n):
            displace, _, stress, _, _, weights = self.Truss25bar(A[ii, :], E, Rho)

            fx[ii, 0] = -weights  # Negate for maximizing optimization

            # Max stress less than 40ksi
            for ss in range(25):
                gx[ii, ss] = abs(stress[ss]) - 40000

            # Max displacement in x and y direction less than .35 inches
            for dd in range(6):
                # print(displace[dd])
                gx[ii, 25 + dd] = max(abs(displace[dd][0]), abs(displace[dd][1])) - 0.35

        return gx, fx
