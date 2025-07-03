import numpy as np
import torch

from ..base import BenchmarkProblem, DataType


class Truss200D(BenchmarkProblem):
    """
    Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
    "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms
    for constrained design optimization of discrete truss structures,"
    Computers & Structures, Volume 212.
    """

    # 200D objective, 200 constraints, X = n-by-200

    available_dimensions = 200
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 200

    tags = {"single_objective", "constrained", "200D", "extra_imports"}

    def __init__(self):
        super().__init__(
            dim=200, num_objectives=1, num_constraints=200, bounds=[(0.1, 33.7)] * 200
        )

    def _evaluate_implementation(self, X, scaling=True, version=1):
        """
        Version [int]: 1, 2, or 3 for different loading conditions:
            (1) 11 nodes with 1e3 N in x direction
            (2) 75 nodes with -1e4 N in y direction
            (3) 11 nodes with 1e3 N in x direction and 75 nodes with -1e4 N in y direction
        """

        if scaling:
            X = super().scale(X)

        # import slientruss3d
        from slientruss3d.truss import Truss
        from slientruss3d.type import MemberType, SupportType

        def Truss200bar(A, E, Rho, version=1):
            # -------------------- Global variables --------------------
            # TEST_OUTPUT_FILE    = f"./test_output.json"
            TRUSS_DIMENSION = 2
            # ----------------------------------------------------------

            # Truss object:
            truss = Truss(dim=TRUSS_DIMENSION)

            # Truss settings (77 joints, 200 members):
            l1 = 240
            l2 = 144
            l3 = 360
            joints = []
            for row in range(11):
                if row % 2 == 0:
                    joints.extend([[col * l1, row * l2] for col in range(5)])
                else:
                    joints.extend([[col * l1 / 2, row * l2] for col in range(9)])
            joints.append([l1, 10 * l2 + l3])
            joints.append([3 * l1, 10 * l2 + l3])

            supports = [SupportType.NO for _ in range(75)] + [
                SupportType.PIN,
                SupportType.PIN,
            ]

            nodes1 = [0, 5, 14, 19, 28, 33, 42, 47, 56, 61, 70]
            nodes2 = [
                0,
                1,
                2,
                3,
                4,
                5,
                7,
                9,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                21,
                23,
                25,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                35,
                37,
                39,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                49,
                51,
                53,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                63,
                65,
                67,
                69,
                70,
                71,
                72,
                73,
                74,
            ]
            if version == 1:
                forces = [[i, (1e3, 0)] for i in nodes1]
            elif version == 2:
                forces = [[i, (0, -1e4)] for i in nodes2]
            elif version == 3:
                forces = [[i, (1e3, 0)] for i in nodes1] + [
                    [i, (0, -1e4)] for i in nodes2
                ]

            members = []
            j_idx = 0
            row_members = np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [0, 5],
                    [0, 6],
                    [1, 6],
                    [1, 7],
                    [1, 8],
                    [2, 8],
                    [2, 9],
                    [2, 10],
                    [3, 10],
                    [3, 11],
                    [3, 12],
                    [4, 12],
                    [4, 13],
                    [5, 6],
                    [6, 7],
                    [7, 8],
                    [8, 9],
                    [9, 10],
                    [10, 11],
                    [11, 12],
                    [12, 13],
                    [5, 14],
                    [6, 14],
                    [6, 15],
                    [7, 15],
                    [8, 15],
                    [8, 16],
                    [9, 16],
                    [10, 16],
                    [10, 17],
                    [11, 17],
                    [12, 17],
                    [12, 18],
                    [13, 18],
                ]
            )
            for row in range(5):
                # 38 each row
                members.extend((row_members + j_idx).tolist())
                j_idx += 14
            members.extend([[70 + i, 71 + i] for i in range(4)])
            members.extend([[70, 75], [71, 75], [72, 75], [72, 76], [73, 76], [74, 76]])

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

            # TrussPlotter(truss).Plot()

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

        if X.size(1) == 200:
            A = X
        elif X.size(1) == 29:
            # Bars in 29 groups because of symmetry
            # (1) A1-A4, (2) A5/8/11/14/17, (3) A19/20/21/22/23/24, (4) A18/25/56/63/94/101/132/139/170/177,
            # (5) A26/29/32/35/38, (6) A6/7/9/10/12/13/15/16/27/28/30/31/33/34/36/37, (7) A39/40/41/42, (8) A43/46/49/52/55,
            # (9) A57/58/59/60/61/62, (10) A64/67/70/73/76, (11) A44/45/47/48/50/51/53/54/65/66/68/69/71/72/74/75,
            # (12) A77/78/79/80, (13) A81/84/87/90/93, (14) A95/96/97/98/99/100, (15) A102/105/108/111/114,
            # (16) A82/83/85/86/88/89/91/92/103/104/106/107/109/110/112/113, (17) A115/116/117/118, (18) A119/122/125/128/131,
            # (19) A133/134/135/136/137/138, (20) A140/143/146/149/152, (21) A120/121/123/124/126/127/129/130/141/142/144/145/147/148/150/151,
            # (22) A153/154/155/156, (23) A157/160/163/166/169, (24) A171/172/173/174/175/176, (25) A178/181/184/187/190,
            # (26) A158/159/161/162/164/165/167/168/179/180/182/183/185/186/188/189, (27) A191/192/193/194, (28) A195/197/198/200, (29) A196/199

            A = torch.zeros(X.size(0), 200)
            A[:, 0:4] = X[:, 0]
            A[:, [4, 7, 10, 13, 16]] = X[:, 1]
            A[:, [18, 19, 20, 21, 22, 23]] = X[:, 2]
            A[:, [17, 24, 55, 62, 93, 100, 131, 138, 169, 176]] = X[:, 3]
            A[:, [25, 28, 31, 34, 37]] = X[:, 4]
            A[:, [5, 6, 8, 9, 11, 12, 14, 15, 26, 27, 29, 30, 32, 33, 35, 36]] = X[:, 5]
            A[:, [38, 39, 40, 41]] = X[:, 6]
            A[:, [42, 45, 48, 51, 54]] = X[:, 7]
            A[:, [56, 57, 58, 59, 60, 61]] = X[:, 8]
            A[:, [63, 66, 69, 72, 75]] = X[:, 9]
            A[:, [43, 44, 46, 47, 49, 50, 52, 53, 64, 65, 67, 68, 70, 71, 73, 74]] = X[
                :, 10
            ]
            A[:, [76, 77, 78, 79]] = X[:, 11]
            A[:, [80, 83, 86, 89, 92]] = X[:, 12]
            A[:, [94, 95, 96, 97, 98, 99]] = X[:, 13]
            A[:, [101, 104, 107, 110, 113]] = X[:, 14]
            A[
                :,
                [
                    81,
                    82,
                    84,
                    85,
                    87,
                    88,
                    90,
                    91,
                    102,
                    103,
                    105,
                    106,
                    108,
                    109,
                    111,
                    112,
                ],
            ] = X[:, 15]
            A[:, [114, 115, 116, 117]] = X[:, 16]
            A[:, [118, 121, 124, 127, 130]] = X[:, 17]
            A[:, [132, 133, 134, 135, 136, 137]] = X[:, 18]
            A[:, [139, 142, 145, 148, 151]] = X[:, 19]
            A[
                :,
                [
                    119,
                    120,
                    122,
                    123,
                    125,
                    126,
                    128,
                    129,
                    140,
                    141,
                    143,
                    144,
                    146,
                    147,
                    149,
                    150,
                ],
            ] = X[:, 20]
            A[:, [152, 153, 154, 155]] = X[:, 21]
            A[:, [156, 159, 162, 165, 168]] = X[:, 22]
            A[:, [170, 171, 172, 173, 174, 175]] = X[:, 23]
            A[:, [177, 180, 183, 186, 189]] = X[:, 24]
            A[
                :,
                [
                    157,
                    158,
                    160,
                    161,
                    163,
                    164,
                    166,
                    167,
                    178,
                    179,
                    181,
                    182,
                    184,
                    185,
                    187,
                    188,
                ],
            ] = X[:, 25]
            A[:, [190, 191, 192, 193]] = X[:, 26]
            A[:, [194, 196, 197, 199]] = X[:, 27]
            A[:, [195, 198]] = X[:, 28]

        E = 3e4 * torch.ones(200)
        Rho = 0.283 * torch.ones(200)

        n = A.size(0)

        fx = torch.zeros(n, 1)

        # 200 bar stress constraints
        gx = torch.zeros(n, 200)

        for ii in range(n):
            displace, _, stress, _, _, weights = Truss200bar(A[ii, :], E, Rho, version)

            fx[ii, 0] = -weights  # Negate for maximizing optimization

            # Max stress less than 10000
            for ss in range(200):
                if ss in stress:
                    gx[ii, ss] = abs(stress[ss]) - 10000
                else:
                    gx[ii, ss] = -10000

        return gx, fx
