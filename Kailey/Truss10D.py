import torch
from base import BenchmarkProblem

class Truss10D(BenchmarkProblem):

    r'''
    Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
    "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms
    for constrained design optimization of discrete truss structures,"
    Computers & Structures, Volume 212.
    '''

    # 10D objective, 14 constraints, X = n-by-10

    tags = {"single_objective", "constrained", "10D", "extra_imports"}

    def __init__(self):
        super().__init__(dim = 10, num_obj = 1, num_cons = 14, bounds = [[0.1, 35]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        # import slientruss3d
        from slientruss3d.truss import Truss
        from slientruss3d.type  import SupportType, MemberType

        def Truss10bar(A, E, Rho):
            # -------------------- Global variables --------------------
            # TEST_OUTPUT_FILE    = f"./test_output.json"
            TRUSS_DIMENSION     = 2
            # ----------------------------------------------------------

            # Truss object:
            truss = Truss(dim=TRUSS_DIMENSION)

            init_truss = truss

            # Truss settings:
            joints = [(720, 360), (720, 0), (360, 360), (360, 0), (0, 360), (0, 0)]
            supports = [SupportType.NO, SupportType.NO, SupportType.NO, SupportType.NO, SupportType.PIN, SupportType.PIN]
            forces = [(1, (0, -1e5)), (3, (0, -1e5))]
            members = [(2, 4), (0, 2), (3, 5), (1, 3), (2, 3), (0, 1), (3, 4), (2, 5), (1, 2), (0, 3)]


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
                # Default: 0.1, 1e7, 1

                memberType = MemberType(A[index].item(), 10000000.0, 0.1)

                if (E != None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), E[index].item(), Rho[index].item())
                elif (E != None) & (Rho==None):
                    memberType = MemberType(A[index].item(), E[index].item(), 0.1)
                elif (E == None) & (Rho!=None):
                    memberType = MemberType(A[index].item(), 10000000.0, Rho[index].item())

                truss.AddNewMember(jointID0, jointID1, memberType)
                index += 1

            # Do direct stiffness method:
            truss.Solve()

            # Dump all the structural analysis results into a .json file:
            # truss.DumpIntoJSON(TEST_OUTPUT_FILE)

            # Get result of structural analysis:
            displace, forces, stress, resistance = truss.GetDisplacements(), truss.GetInternalForces(), truss.GetInternalStresses(), truss.GetResistances()
            return displace, forces, stress, resistance, truss, truss.weight

        E = 1e7 * torch.ones(10)
        Rho = 0.1 * torch.ones(10)

        n = X.size(0)

        fx = torch.zeros(n,1)

        # 10 bar stress constraints, 4 displacement constraints
        gx = torch.zeros(n,14)

        for ii in range(n):

            displace, _, stress, _, _, weights = Truss10bar(X[ii,:], E, Rho)

            fx[ii,0] = -weights           # Negate for maximizing optimization

            for ss in range(10):
                gx[ii,ss] = abs(stress[ss]) - 25000

            gx[ii,10] = max(abs(displace[0][0]), abs(displace[0][1])) - 2
            gx[ii,11] = max(abs(displace[1][0]), abs(displace[1][1])) - 2
            gx[ii,12] = max(abs(displace[2][0]), abs(displace[2][1])) - 2
            gx[ii,13] = max(abs(displace[3][0]), abs(displace[3][1])) - 2

        return gx, fx
