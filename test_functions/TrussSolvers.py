import matplotlib.pyplot as plt
import torch
import numpy as np

from slientruss3d.truss import Truss
from slientruss3d.type  import SupportType, MemberType
from slientruss3d.plot  import TrussPlotter



#
#
#   A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms for constrained design optimization of discrete truss structures
#
#   Reference (Mixed 10-bar, 25-bar, 72-bar):
#
#     Duc Thang Le, Dac-Khuong Bui, Tuan Duc Ngo, Quoc-Hung Nguyen, H. Nguyen-Xuan, (2019).
#     "A novel hybrid method combining electromagnetism-like mechanism and firefly algorithms 
#     for constrained design optimization of discrete truss structures," 
#     Computers & Structures, Volume 212.
#
#


def TestPlot(truss):
    # -------------------- Global variables --------------------
    # Files settings:
    TEST_FILE_NUMBER        = 25
    TEST_LOAD_CASE          = 0
    TEST_INPUT_FILE         = f"./data/bar-{TEST_FILE_NUMBER}_output_{TEST_LOAD_CASE}.json"
    TEST_PLOT_SAVE_PATH     = f"./plot/bar-{TEST_FILE_NUMBER}_plot_{TEST_LOAD_CASE}.png"

    # Truss dimension setting:
    TRUSS_DIMENSION         = 3

    # Figure layout settings:
    IS_SAVE_PLOT            = False   # Whether to save truss figure or not.
    IS_EQUAL_AXIS           = True    # Whether to use actual aspect ratio in the truss figure or not.
    IS_PLOT_STRESS          = True    # If True, the color of each displaced member gives expression to [stress]. Otherwise, [force magnitude].
    MAX_SCALED_DISPLACEMENT = 15      # Scale the max value of all dimensions of displacements.
    MAX_SCALED_FORCE        = 50      # Scale the max value of all dimensions of force arrows.
    POINT_SIZE_SCALE_FACTOR = 1       # Scale the default size of joint point in the truss figure.
    ARROW_SIZE_SCALE_FACTOR = 1       # Scale the default size of force arrow in the truss figure.
    # ----------------------------------------------------------

    # Truss object:
    # truss = Truss(dim=TRUSS_DIMENSION)

    # You could directly read the output .json file.
    # truss.LoadFromJSON(TEST_INPUT_FILE, isOutputFile=True)

    # Show or save the structural analysis result figure:
    TrussPlotter(truss,
                 isEqualAxis=IS_EQUAL_AXIS,
                 isPlotStress=IS_PLOT_STRESS,
                 maxScaledDisplace=MAX_SCALED_DISPLACEMENT, 
                 maxScaledForce=MAX_SCALED_FORCE,
                 pointScale=POINT_SIZE_SCALE_FACTOR,
                 arrowScale=ARROW_SIZE_SCALE_FACTOR).Plot(IS_SAVE_PLOT, TEST_PLOT_SAVE_PATH)














def Truss10bar(A, E, Rho):
    # -------------------- Global variables --------------------
    TEST_OUTPUT_FILE    = f"./test_output.json"
    TRUSS_DIMENSION     = 2
    # ----------------------------------------------------------

    # Truss object:
    truss = Truss(dim=TRUSS_DIMENSION)

    init_truss = truss

    # Truss settings:
    joints = [
        (720, 360),
        (720, 0),
        (360, 360),
        (360, 0),
        (0, 360),
        (0, 0)
    ]
    

    supports = [
        SupportType.NO,
        SupportType.NO,
        SupportType.NO,
        SupportType.NO,
        SupportType.PIN,
        SupportType.PIN
    ]
    
    forces = [(1, (0, -1e5)), (3, (0, -1e5))]
    
    members = [
        (2, 4),
        (0, 2),
        (3, 5),
        (1, 3),
        (2, 3),
        (0, 1),
        (3, 4),
        (2, 5),
        (1, 2),
        (0, 3)
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






def Truss25bar(A, E, Rho):
    # -------------------- Global variables --------------------
    TEST_OUTPUT_FILE    = f"./test_output.json"
    TRUSS_DIMENSION     = 3
    # ----------------------------------------------------------

    # Truss object:
    truss = Truss(dim=TRUSS_DIMENSION)

    init_truss = truss

    # Truss settings:
    joints     = [(62.5, 100, 200), 
                  (137.5, 100, 200), 
                  (62.5, 137.5, 100), 
                  (137.5, 137.5, 100), 
                  (137.5, 62.5, 100),
                  (62.5, 62.5, 100),
                  (0, 200, 0),
                  (200, 200, 0),
                  (200, 0, 0),
                  (0, 0, 0)
                 ]
    
    
    supports   = [SupportType.NO, 
                  SupportType.NO, 
                  SupportType.NO, 
                  SupportType.NO, 
                  SupportType.NO,
                  SupportType.NO, 
                  SupportType.PIN,
                  SupportType.PIN,
                  SupportType.PIN,
                  SupportType.PIN
                 ]
    forces     = [(0, (1000, -10000, -10000)), 
                  (1, (0, -10000, -10000)),
                  (2, (500, 0, 0)),
                  (5, (600, 0, 0)),
                 ]
    
    
    members    = [(0, 1), 
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
        
        memberType = MemberType(A[index].item(), 3e7, .283)

        if (E != None) & (Rho!=None):
            memberType = MemberType(A[index].item(), E[index].item(), Rho[index].item())
        elif (E != None) & (Rho==None):
            memberType = MemberType(A[index].item(), E[index].item(), .283)
        elif (E == None) & (Rho!=None):   
            memberType = MemberType(A[index].item(), 3e7, Rho[index].item())



        # memberType = MemberType(A[index].item(), 1e7, .1)
        truss.AddNewMember(jointID0, jointID1, memberType)
        index += 1

    # Do direct stiffness method:
    truss.Solve()

    # Dump all the structural analysis results into a .json file:
    # truss.DumpIntoJSON(TEST_OUTPUT_FILE)
    
    # Get result of structural analysis:
    displace, forces, stress, resistance = truss.GetDisplacements(), truss.GetInternalForces(), truss.GetInternalStresses(), truss.GetResistances()
    return displace, forces, stress, resistance, truss, truss.weight








def Truss47bar(A, E, Rho, version):
    # -------------------- Global variables --------------------
    TEST_OUTPUT_FILE    = f"./test_output.json"
    TRUSS_DIMENSION     = 2
    # ----------------------------------------------------------

    # Truss object:
    truss = Truss(dim=TRUSS_DIMENSION)

    init_truss = truss

    # Truss settings:
    joints = [(-60, 0), 
          (60, 0), 
          (-60, 120), 
          (60, 120), 
          (-60, 240), 
          (60, 240), 
          (-60, 360), 
          (60, 360), 
          (-30, 420), 
          (30, 420), 
          (-30, 480), 
          (30, 480), 
          (-30, 540), 
          (30, 540), 
          (-90, 570), 
          (90, 570), 
          (-150, 600), 
          (-90, 600), 
          (-30, 600), 
          (30, 600), 
          (90, 600), 
          (150, 600)]

    supports = [SupportType.PIN, SupportType.PIN,
                SupportType.NO, SupportType.NO, 
                SupportType.NO, SupportType.NO, 
                SupportType.NO, SupportType.NO, 
                SupportType.NO, SupportType.NO, 
                SupportType.NO, SupportType.NO, 
                SupportType.NO, SupportType.NO, 
                SupportType.NO, SupportType.NO, 
                SupportType.NO, SupportType.NO, 
                SupportType.NO, SupportType.NO, 
                SupportType.NO, SupportType.NO]
    

    forces = [(16, (6000, -14000)), 
              (21, (6000, -14000))]
    
    if version == 1:
        forces = [(16, (6000, -14000)), 
                  (21, (6000, -14000))]
        
    elif  version == 2:
        forces = [(16, (6000, -14000))]
        
    elif  version == 3:   
        forces = [(21, (6000, -14000))]
        
    
    
    
    members = [(0, 2), (0, 3), (1, 2), (1, 3), 
               (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), 
               (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), 
               (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9), 
               (8, 10), (8, 11), (9, 10), (9, 11), 
               (10, 11), (10, 12), (10, 13), (11, 12), (11, 13), 
               (12, 13), (12, 14), (12, 18), (12, 19), 
               (13, 18), (13, 19), (13, 15), (14, 16), 
               (14, 17), (14, 18), (15, 19), (15, 20), 
               (15, 21), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21)]
    
    
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
        # memberType = MemberType(A[index].item(), 300000000.0, 0.3)


        # Default: 0.1, 300000000.0, 0.3
        
        memberType = MemberType(A[index].item(), 300000000.0, 0.3)

        if (E != None) & (Rho!=None):
            memberType = MemberType(A[index].item(), E[index].item(), Rho[index].item())
        elif (E != None) & (Rho==None):
            memberType = MemberType(A[index].item(), E[index].item(), 0.3)
        elif (E == None) & (Rho!=None):   
            memberType = MemberType(A[index].item(), 300000000.0, Rho[index].item())


        truss.AddNewMember(jointID0, jointID1, memberType)
        index += 1

    # Do direct stiffness method:
    truss.Solve()

    # Dump all the structural analysis results into a .json file:
    # truss.DumpIntoJSON(TEST_OUTPUT_FILE)
    
    # Get result of structural analysis:
    displace, stress, resistance = truss.GetDisplacements(), truss.GetInternalStresses(), truss.GetResistances()
    return displace, stress, resistance, truss, truss.weight







def Truss120bar(A, E, Rho):
    # -------------------- Global variables --------------------
    TEST_OUTPUT_FILE    = f"./test_output.json"
    TRUSS_DIMENSION     = 3
    # ----------------------------------------------------------

    # Truss object:
    truss = Truss(dim=TRUSS_DIMENSION)

    init_truss = truss

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
        (541.7768323535071, -312.79499999999996, 0.0)
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
        SupportType.PIN
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
        (36, (0.0, 0.0, -2248.0))
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
            (36, 37)
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

        if (E != None) & (Rho!=None):
            memberType = MemberType(A[index].item(), E[index].item(), Rho[index].item())
        elif (E != None) & (Rho==None):
            memberType = MemberType(A[index].item(), E[index].item(), 0.288)
        elif (E == None) & (Rho!=None):   
            memberType = MemberType(A[index].item(), 30450000, Rho[index].item())



        truss.AddNewMember(jointID0, jointID1, memberType)
        index += 1

    # Do direct stiffness method:
    truss.Solve()

    # Dump all the structural analysis results into a .json file:
    # truss.DumpIntoJSON(TEST_OUTPUT_FILE)
    
    # Get result of structural analysis:
    displace, forces, stress, resistance = truss.GetDisplacements(), truss.GetInternalForces(), truss.GetInternalStresses(), truss.GetResistances()
    return displace, forces, stress, resistance, truss, truss.weight






def Truss72bar(A, E, Rho, version="4_forces"):
    # -------------------- Global variables --------------------
    TEST_OUTPUT_FILE    = f"./test_output.json"
    TRUSS_DIMENSION     = 3
    # ----------------------------------------------------------

    # Truss object:
    truss = Truss(dim=TRUSS_DIMENSION)

    init_truss = truss

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
            (0.0, 120.0, 240.0)
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
            SupportType.NO
        ]




    if version == "4_forces":
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
                (19, (0.0, 0.0, -5000.0))
            ]
    elif version == "single":
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
            (19, (0.0, 0.0, 0.0))
        ]
    else:
        raise ValueError("Invalid version, choose between '4_forces' and 'single'")


    
    
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
            (17, 19)
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



def Truss200bar(A, E, Rho, version=1):
    # -------------------- Global variables --------------------
    TEST_OUTPUT_FILE    = f"./test_output.json"
    TRUSS_DIMENSION     = 2
    # ----------------------------------------------------------

    # Truss object:
    truss = Truss(dim=TRUSS_DIMENSION)

    init_truss = truss

    # Truss settings (77 joints, 200 members):
    l1 = 240
    l2 = 144
    l3 = 360
    joints = []
    for row in range(11):
        if row % 2 == 0:
            joints.extend([[col*l1, row*l2] for col in range(5)])
        else:
            joints.extend([[col*l1/2, row*l2] for col in range(9)])
    joints.append([l1, 10*l2+l3])
    joints.append([3*l1, 10*l2+l3])


    supports = [SupportType.NO for _ in range(75)] + [SupportType.PIN, SupportType.PIN]


    nodes1 = [0, 5, 14, 19, 28, 33, 42, 47, 56, 61, 70]
    nodes2 = [
        0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 21, 23, 25,
        27, 28, 29, 30, 31, 32, 33, 35, 37, 39, 41, 42, 43, 44, 45, 46, 47, 49,
        51, 53, 55, 56, 57, 58, 59, 60, 61, 63, 65, 67, 69, 70, 71, 72, 73, 74
    ]
    if version == 1:
        forces = [[i, (1e3, 0)] for i in nodes1]
    elif version == 2:
        forces = [[i, (0, -1e4)] for i in nodes2]
    elif version == 3:
        forces = [[i, (1e3, 0)] for i in nodes1] + [[i, (0, -1e4)] for i in nodes2]

    
    members = []
    j_idx = 0
    row_members = np.array([
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [0, 6], [1, 6], [1, 7], [1, 8], [2, 8], [2, 9], [2, 10],
        [3, 10], [3, 11], [3, 12], [4, 12], [4, 13],
        [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13],
        [5, 14], [6, 14], [6, 15], [7, 15], [8, 15], [8, 16], [9, 16], [10, 16],
        [10, 17], [11, 17], [12, 17], [12, 18], [13, 18],
    ])
    for row in range(5):
        # 38 each row
        members.extend((row_members + j_idx).tolist())
        j_idx += 14
    members.extend([[70+i, 71+i] for i in range(4)])
    members.extend([
        [70, 75], [71, 75], [72, 75], [72, 76], [73, 76], [74, 76]
    ])



    for joint, support in zip(joints, supports):
        truss.AddNewJoint(joint, support)
        
    for jointID, force in forces:
        truss.AddExternalForce(jointID, force)
    
    index = 0
    for jointID0, jointID1 in members:
        # memberType = MemberType(A[index].item(), 10000000.0, 0.1)

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

    # TrussPlotter(truss).Plot()

    # Dump all the structural analysis results into a .json file:
    # truss.DumpIntoJSON(TEST_OUTPUT_FILE)
    
    # Get result of structural analysis:
    displace, forces, stress, resistance = truss.GetDisplacements(), truss.GetInternalForces(), truss.GetInternalStresses(), truss.GetResistances()
    return displace, forces, stress, resistance, truss, truss.weight


        





def Truss942bar(A, E, Rho):
    # -------------------- Global variables --------------------
    TEST_OUTPUT_FILE    = f"./test_output.json"
    TRUSS_DIMENSION     = 3
    # ----------------------------------------------------------

    # Truss object:
    truss = Truss(dim=TRUSS_DIMENSION)

    init_truss = truss

    # Truss settings:
    joints = [
        [24.5, 24.5, 312.0],
        [38.5, 24.5, 312.0],
        [24.5, 38.5, 312.0],
        [38.5, 38.5, 312.0],
        [24.5, 24.5, 300.0],
        [38.5, 24.5, 300.0],
        [24.5, 38.5, 300.0],
        [38.5, 38.5, 300.0],
        [24.5, 24.5, 288.0],
        [38.5, 24.5, 288.0],
        [24.5, 38.5, 288.0],
        [38.5, 38.5, 288.0],
        [24.5, 24.5, 276.0],
        [38.5, 24.5, 276.0],
        [24.5, 38.5, 276.0],
        [38.5, 38.5, 276.0],
        [24.5, 24.5, 264.0],
        [38.5, 24.5, 264.0],
        [24.5, 38.5, 264.0],
        [38.5, 38.5, 264.0],
        [24.5, 24.5, 252.0],
        [38.5, 24.5, 252.0],
        [24.5, 38.5, 252.0],
        [38.5, 38.5, 252.0],
        [21.6005, 21.6005, 240.0],
        [31.5, 17.5, 240.0],
        [41.3995, 21.6005, 240.0],
        [17.5, 31.5, 240.0],
        [45.5, 31.5, 240.0],
        [21.6005, 41.3995, 240.0],
        [31.5, 45.5, 240.0],
        [41.3995, 41.3995, 240.0],
        [21.6005, 21.6005, 228.0],
        [31.5, 17.5, 228.0],
        [41.3995, 21.6005, 228.0],
        [17.5, 31.5, 228.0],
        [45.5, 31.5, 228.0],
        [21.6005, 41.3995, 228.0],
        [31.5, 45.5, 228.0],
        [41.3995, 41.3995, 228.0],
        [21.6005, 21.6005, 216.0],
        [31.5, 17.5, 216.0],
        [41.3995, 21.6005, 216.0],
        [17.5, 31.5, 216.0],
        [45.5, 31.5, 216.0],
        [21.6005, 41.3995, 216.0],
        [31.5, 45.5, 216.0],
        [41.3995, 41.3995, 216.0],
        [21.6005, 21.6005, 204.0],
        [31.5, 17.5, 204.0],
        [41.3995, 21.6005, 204.0],
        [17.5, 31.5, 204.0],
        [45.5, 31.5, 204.0],
        [21.6005, 41.3995, 204.0],
        [31.5, 45.5, 204.0],
        [41.3995, 41.3995, 204.0],
        [21.6005, 21.6005, 192.0],
        [31.5, 17.5, 192.0],
        [41.3995, 21.6005, 192.0],
        [17.5, 31.5, 192.0],
        [45.5, 31.5, 192.0],
        [21.6005, 41.3995, 192.0],
        [31.5, 45.5, 192.0],
        [41.3995, 41.3995, 192.0],
        [21.6005, 21.6005, 180.0],
        [31.5, 17.5, 180.0],
        [41.3995, 21.6005, 180.0],
        [17.5, 31.5, 180.0],
        [45.5, 31.5, 180.0],
        [21.6005, 41.3995, 180.0],
        [31.5, 45.5, 180.0],
        [41.3995, 41.3995, 180.0],
        [21.6005, 21.6005, 168.0],
        [31.5, 17.5, 168.0],
        [41.3995, 21.6005, 168.0],
        [17.5, 31.5, 168.0],
        [45.5, 31.5, 168.0],
        [21.6005, 41.3995, 168.0],
        [31.5, 45.5, 168.0],
        [41.3995, 41.3995, 168.0],
        [21.6005, 21.6005, 156.0],
        [31.5, 17.5, 156.0],
        [41.3995, 21.6005, 156.0],
        [17.5, 31.5, 156.0],
        [45.5, 31.5, 156.0],
        [21.6005, 41.3995, 156.0],
        [31.5, 45.5, 156.0],
        [41.3995, 41.3995, 156.0],
        [7.0, 24.5, 144.0],
        [11.1199, 11.1199, 144.0],
        [24.5, 7.0, 144.0],
        [38.5, 7.0, 144.0],
        [51.8801, 11.1199, 144.0],
        [56.0, 24.5, 144.0],
        [7.0, 38.5, 144.0],
        [11.1199, 51.8801, 144.0],
        [24.5, 56.0, 144.0],
        [38.5, 56.0, 144.0],
        [51.8801, 51.8801, 144.0],
        [56.0, 38.5, 144.0],
        [7.0, 24.5, 132.0],
        [11.1199, 11.1199, 132.0],
        [24.5, 7.0, 132.0],
        [38.5, 7.0, 132.0],
        [51.8801, 11.1199, 132.0],
        [56.0, 24.5, 132.0],
        [7.0, 38.5, 132.0],
        [11.1199, 51.8801, 132.0],
        [24.5, 56.0, 132.0],
        [38.5, 56.0, 132.0],
        [51.8801, 51.8801, 132.0],
        [56.0, 38.5, 132.0],
        [7.0, 24.5, 120.0],
        [11.1199, 11.1199, 120.0],
        [24.5, 7.0, 120.0],
        [38.5, 7.0, 120.0],
        [51.8801, 11.1199, 120.0],
        [56.0, 24.5, 120.0],
        [7.0, 38.5, 120.0],
        [11.1199, 51.8801, 120.0],
        [24.5, 56.0, 120.0],
        [38.5, 56.0, 120.0],
        [51.8801, 51.8801, 120.0],
        [56.0, 38.5, 120.0],
        [7.0, 24.5, 108.0],
        [11.1199, 11.1199, 108.0],
        [24.5, 7.0, 108.0],
        [38.5, 7.0, 108.0],
        [51.8801, 11.1199, 108.0],
        [56.0, 24.5, 108.0],
        [7.0, 38.5, 108.0],
        [11.1199, 51.8801, 108.0],
        [24.5, 56.0, 108.0],
        [38.5, 56.0, 108.0],
        [51.8801, 51.8801, 108.0],
        [56.0, 38.5, 108.0],
        [7.0, 24.5, 96.0],
        [11.1199, 11.1199, 96.0],
        [24.5, 7.0, 96.0],
        [38.5, 7.0, 96.0],
        [51.8801, 11.1199, 96.0],
        [56.0, 24.5, 96.0],
        [7.0, 38.5, 96.0],
        [11.1199, 51.8801, 96.0],
        [24.5, 56.0, 96.0],
        [38.5, 56.0, 96.0],
        [51.8801, 51.8801, 96.0],
        [56.0, 38.5, 96.0],
        [7.0, 24.5, 84.0],
        [11.1199, 11.1199, 84.0],
        [24.5, 7.0, 84.0],
        [38.5, 7.0, 84.0],
        [51.8801, 11.1199, 84.0],
        [56.0, 24.5, 84.0],
        [7.0, 38.5, 84.0],
        [11.1199, 51.8801, 84.0],
        [24.5, 56.0, 84.0],
        [38.5, 56.0, 84.0],
        [51.8801, 51.8801, 84.0],
        [56.0, 38.5, 84.0],
        [7.0, 24.5, 72.0],
        [11.1199, 11.1199, 72.0],
        [24.5, 7.0, 72.0],
        [38.5, 7.0, 72.0],
        [51.8801, 11.1199, 72.0],
        [56.0, 24.5, 72.0],
        [7.0, 38.5, 72.0],
        [11.1199, 51.8801, 72.0],
        [24.5, 56.0, 72.0],
        [38.5, 56.0, 72.0],
        [51.8801, 51.8801, 72.0],
        [56.0, 38.5, 72.0],
        [7.0, 24.5, 60.0],
        [11.1199, 11.1199, 60.0],
        [24.5, 7.0, 60.0],
        [38.5, 7.0, 60.0],
        [51.8801, 11.1199, 60.0],
        [56.0, 24.5, 60.0],
        [7.0, 38.5, 60.0],
        [11.1199, 51.8801, 60.0],
        [24.5, 56.0, 60.0],
        [38.5, 56.0, 60.0],
        [51.8801, 51.8801, 60.0],
        [56.0, 38.5, 60.0],
        [7.0, 24.5, 48.0],
        [11.1199, 11.1199, 48.0],
        [24.5, 7.0, 48.0],
        [38.5, 7.0, 48.0],
        [51.8801, 11.1199, 48.0],
        [56.0, 24.5, 48.0],
        [7.0, 38.5, 48.0],
        [11.1199, 51.8801, 48.0],
        [24.5, 56.0, 48.0],
        [38.5, 56.0, 48.0],
        [51.8801, 51.8801, 48.0],
        [56.0, 38.5, 48.0],
        [7.0, 24.5, 36.0],
        [11.1199, 11.1199, 36.0],
        [24.5, 7.0, 36.0],
        [38.5, 7.0, 36.0],
        [51.8801, 11.1199, 36.0],
        [56.0, 24.5, 36.0],
        [7.0, 38.5, 36.0],
        [11.1199, 51.8801, 36.0],
        [24.5, 56.0, 36.0],
        [38.5, 56.0, 36.0],
        [51.8801, 51.8801, 36.0],
        [56.0, 38.5, 36.0],
        [7.0, 24.5, 24.0],
        [11.1199, 11.1199, 24.0],
        [24.5, 7.0, 24.0],
        [38.5, 7.0, 24.0],
        [51.8801, 11.1199, 24.0],
        [56.0, 24.5, 24.0],
        [7.0, 38.5, 24.0],
        [11.1199, 51.8801, 24.0],
        [24.5, 56.0, 24.0],
        [38.5, 56.0, 24.0],
        [51.8801, 51.8801, 24.0],
        [56.0, 38.5, 24.0],
        [7.0, 24.5, 12.0],
        [11.1199, 11.1199, 12.0],
        [24.5, 7.0, 12.0],
        [38.5, 7.0, 12.0],
        [51.8801, 11.1199, 12.0],
        [56.0, 24.5, 12.0],
        [7.0, 38.5, 12.0],
        [11.1199, 51.8801, 12.0],
        [24.5, 56.0, 12.0],
        [38.5, 56.0, 12.0],
        [51.8801, 51.8801, 12.0],
        [56.0, 38.5, 12.0],
        [7.0, 24.5, 0.0],
        [11.1199, 11.1199, 0.0],
        [24.5, 7.0, 0.0],
        [38.5, 7.0, 0.0],
        [51.8801, 11.1199, 0.0],
        [56.0, 24.5, 0.0],
        [7.0, 38.5, 0.0],
        [11.1199, 51.8801, 0.0],
        [24.5, 56.0, 0.0],
        [38.5, 56.0, 0.0],
        [51.8801, 51.8801, 0.0],
        [56.0, 38.5, 0.0]
      ]


    supports = []
    for ii in range(244-12):
        supports.append(SupportType.NO)
    for ii in range(12):
        supports.append(SupportType.PIN)

    
    forces = [
            (0, (1.5, -1.0, -3.0)),
            (1, (-1.0, -1.0, -3.0)),
            (2, (1.5, -1.0, -3.0)),
            (3, (-1.0, 1.0, -3.0)),
            (4, (1.5, -1.0, -3.0)),
            (5, (-1.0, -1.0, -3.0)),
            (6, (1.5, -1.0, -3.0)),
            (7, (-1.0, 1.0, -3.0)),
            (8, (1.5, -1.0, -3.0)),
            (9, (-1.0, -1.0, -3.0)),
            (10, (1.5, -1.0, -3.0)),
            (11, (-1.0, 1.0, -3.0)),
            (12, (1.5, -1.0, -3.0)),
            (13, (-1.0, -1.0, -3.0)),
            (14, (1.5, -1.0, -3.0)),
            (15, (-1.0, 1.0, -3.0)),
            (16, (1.5, -1.0, -3.0)),
            (17, (-1.0, -1.0, -3.0)),
            (18, (1.5, -1.0, -3.0)),
            (19, (-1.0, 1.0, -3.0)),
            (20, (1.5, -1.0, -3.0)),
            (21, (-1.0, -1.0, -3.0)),
            (22, (1.5, -1.0, -3.0)),
            (23, (-1.0, 1.0, -3.0)),
            (24, (1.5, -1.0, -3.0)),
            (25, (0.0, -1.0, -3.0)),
            (26, (-1.0, -1.0, -3.0)),
            (27, (1.5, 0.0, -3.0)),
            (28, (-1.0, 0.0, -3.0)),
            (29, (1.5, 1.0, -3.0)),
            (30, (0.0, 1.0, -3.0)),
            (31, (-1.0, 1.0, -3.0)),
            (32, (1.5, -1.0, -6.0)),
            (33, (0.0, -1.0, -6.0)),
            (34, (-1.0, -1.0, -6.0)),
            (35, (1.5, 0.0, -6.0)),
            (36, (-1.0, 0.0, -6.0)),
            (37, (1.5, 1.0, -6.0)),
            (38, (0.0, 1.0, -6.0)),
            (39, (-1.0, 1.0, -6.0)),
            (40, (1.5, -1.0, -6.0)),
            (41, (0.0, -1.0, -6.0)),
            (42, (-1.0, -1.0, -6.0)),
            (43, (1.5, 0.0, -6.0)),
            (44, (-1.0, 0.0, -6.0)),
            (45, (1.5, 1.0, -6.0)),
            (46, (0.0, 1.0, -6.0)),
            (47, (-1.0, 1.0, -6.0)),
            (48, (1.5, -1.0, -6.0)),
            (49, (0.0, -1.0, -6.0)),
            (50, (-1.0, -1.0, -6.0)),
            (51, (1.5, 0.0, -6.0)),
            (52, (-1.0, 0.0, -6.0)),
            (53, (1.5, 1.0, -6.0)),
            (54, (0.0, 1.0, -6.0)),
            (55, (-1.0, 1.0, -6.0)),
            (56, (1.5, -1.0, -6.0)),
            (57, (0.0, -1.0, -6.0)),
            (58, (-1.0, -1.0, -6.0)),
            (59, (1.5, 0.0, -6.0)),
            (60, (-1.0, 0.0, -6.0)),
            (61, (1.5, 1.0, -6.0)),
            (62, (0.0, 1.0, -6.0)),
            (63, (-1.0, 1.0, -6.0)),
            (64, (1.5, -1.0, -6.0)),
            (65, (0.0, -1.0, -6.0)),
            (66, (-1.0, -1.0, -6.0)),
            (67, (1.5, 0.0, -6.0)),
            (68, (-1.0, 0.0, -6.0)),
            (69, (1.5, 1.0, -6.0)),
            (70, (0.0, 1.0, -6.0)),
            (71, (-1.0, 1.0, -6.0)),
            (72, (1.5, -1.0, -6.0)),
            (73, (0.0, -1.0, -6.0)),
            (74, (-1.0, -1.0, -6.0)),
            (75, (1.5, 0.0, -6.0)),
            (76, (-1.0, 0.0, -6.0)),
            (77, (1.5, 1.0, -6.0)),
            (78, (0.0, 1.0, -6.0)),
            (79, (-1.0, 1.0, -6.0)),
            (80, (1.5, -1.0, -6.0)),
            (81, (0.0, -1.0, -6.0)),
            (82, (-1.0, -1.0, -6.0)),
            (83, (1.5, 0.0, -6.0)),
            (84, (-1.0, 0.0, -6.0)),
            (85, (1.5, 1.0, -6.0)),
            (86, (0.0, 1.0, -6.0)),
            (87, (-1.0, 1.0, -6.0)),
            (88, (1.5, -1.0, -6.0)),
            (89, (1.5, -1.0, -6.0)),
            (90, (1.5, -1.0, -6.0)),
            (91, (-1.0, -1.0, -6.0)),
            (92, (-1.0, -1.0, -6.0)),
            (93, (-1.0, -1.0, -6.0)),
            (94, (1.5, 1.0, -6.0)),
            (95, (1.5, 1.0, -6.0)),
            (96, (1.5, 1.0, -6.0)),
            (97, (-1.0, 1.0, -6.0)),
            (98, (-1.0, 1.0, -6.0)),
            (99, (-1.0, 1.0, -6.0)),
            (100, (1.5, -1.0, -9.0)),
            (101, (1.5, -1.0, -9.0)),
            (102, (1.5, -1.0, -9.0)),
            (103, (-1.0, -1.0, -9.0)),
            (104, (-1.0, -1.0, -9.0)),
            (105, (-1.0, -1.0, -9.0)),
            (106, (1.5, 1.0, -9.0)),
            (107, (1.5, 1.0, -9.0)),
            (108, (1.5, 1.0, -9.0)),
            (109, (-1.0, 1.0, -9.0)),
            (110, (-1.0, 1.0, -9.0)),
            (111, (-1.0, 1.0, -9.0)),
            (112, (1.5, -1.0, -9.0)),
            (113, (1.5, -1.0, -9.0)),
            (114, (1.5, -1.0, -9.0)),
            (115, (-1.0, -1.0, -9.0)),
            (116, (-1.0, -1.0, -9.0)),
            (117, (-1.0, -1.0, -9.0)),
            (118, (1.5, 1.0, -9.0)),
            (119, (1.5, 1.0, -9.0)),
            (120, (1.5, 1.0, -9.0)),
            (121, (-1.0, 1.0, -9.0)),
            (122, (-1.0, 1.0, -9.0)),
            (123, (-1.0, 1.0, -9.0)),
            (124, (1.5, -1.0, -9.0)),
            (125, (1.5, -1.0, -9.0)),
            (126, (1.5, -1.0, -9.0)),
            (127, (-1.0, -1.0, -9.0)),
            (128, (-1.0, -1.0, -9.0)),
            (129, (-1.0, -1.0, -9.0)),
            (130, (1.5, 1.0, -9.0)),
            (131, (1.5, 1.0, -9.0)),
            (132, (1.5, 1.0, -9.0)),
            (133, (-1.0, 1.0, -9.0)),
            (134, (-1.0, 1.0, -9.0)),
            (135, (-1.0, 1.0, -9.0)),
            (136, (1.5, -1.0, -9.0)),
            (137, (1.5, -1.0, -9.0)),
            (138, (1.5, -1.0, -9.0)),
            (139, (-1.0, -1.0, -9.0)),
            (140, (-1.0, -1.0, -9.0)),
            (141, (-1.0, -1.0, -9.0)),
            (142, (1.5, 1.0, -9.0)),
            (143, (1.5, 1.0, -9.0)),
            (144, (1.5, 1.0, -9.0)),
            (145, (-1.0, 1.0, -9.0)),
            (146, (-1.0, 1.0, -9.0)),
            (147, (-1.0, 1.0, -9.0)),
            (148, (1.5, -1.0, -9.0)),
            (149, (1.5, -1.0, -9.0)),
            (150, (1.5, -1.0, -9.0)),
            (151, (-1.0, -1.0, -9.0)),
            (152, (-1.0, -1.0, -9.0)),
            (153, (-1.0, -1.0, -9.0)),
            (154, (1.5, 1.0, -9.0)),
            (155, (1.5, 1.0, -9.0)),
            (156, (1.5, 1.0, -9.0)),
            (157, (-1.0, 1.0, -9.0)),
            (158, (-1.0, 1.0, -9.0)),
            (159, (-1.0, 1.0, -9.0)),
            (160, (1.5, -1.0, -9.0)),
            (161, (1.5, -1.0, -9.0)),
            (162, (1.5, -1.0, -9.0)),
            (163, (-1.0, -1.0, -9.0)),
            (164, (-1.0, -1.0, -9.0)),
            (165, (-1.0, -1.0, -9.0)),
            (166, (1.5, 1.0, -9.0)),
            (167, (1.5, 1.0, -9.0)),
            (168, (1.5, 1.0, -9.0)),
            (169, (-1.0, 1.0, -9.0)),
            (170, (-1.0, 1.0, -9.0)),
            (171, (-1.0, 1.0, -9.0)),
            (172, (1.5, -1.0, -9.0)),
            (173, (1.5, -1.0, -9.0)),
            (174, (1.5, -1.0, -9.0)),
            (175, (-1.0, -1.0, -9.0)),
            (176, (-1.0, -1.0, -9.0)),
            (177, (-1.0, -1.0, -9.0)),
            (178, (1.5, 1.0, -9.0)),
            (179, (1.5, 1.0, -9.0)),
            (180, (1.5, 1.0, -9.0)),
            (181, (-1.0, 1.0, -9.0)),
            (182, (-1.0, 1.0, -9.0)),
            (183, (-1.0, 1.0, -9.0)),
            (184, (1.5, -1.0, -9.0)),
            (185, (1.5, -1.0, -9.0)),
            (186, (1.5, -1.0, -9.0)),
            (187, (-1.0, -1.0, -9.0)),
            (188, (-1.0, -1.0, -9.0)),
            (189, (-1.0, -1.0, -9.0)),
            (190, (1.5, 1.0, -9.0)),
            (191, (1.5, 1.0, -9.0)),
            (192, (1.5, 1.0, -9.0)),
            (193, (-1.0, 1.0, -9.0)),
            (194, (-1.0, 1.0, -9.0)),
            (195, (-1.0, 1.0, -9.0)),
            (196, (1.5, -1.0, -9.0)),
            (197, (1.5, -1.0, -9.0)),
            (198, (1.5, -1.0, -9.0)),
            (199, (-1.0, -1.0, -9.0)),
            (200, (-1.0, -1.0, -9.0)),
            (201, (-1.0, -1.0, -9.0)),
            (202, (1.5, 1.0, -9.0)),
            (203, (1.5, 1.0, -9.0)),
            (204, (1.5, 1.0, -9.0)),
            (205, (-1.0, 1.0, -9.0)),
            (206, (-1.0, 1.0, -9.0)),
            (207, (-1.0, 1.0, -9.0)),
            (208, (1.5, -1.0, -9.0)),
            (209, (1.5, -1.0, -9.0)),
            (210, (1.5, -1.0, -9.0)),
            (211, (-1.0, -1.0, -9.0)),
            (212, (-1.0, -1.0, -9.0)),
            (213, (-1.0, -1.0, -9.0)),
            (214, (1.5, 1.0, -9.0)),
            (215, (1.5, 1.0, -9.0)),
            (216, (1.5, 1.0, -9.0)),
            (217, (-1.0, 1.0, -9.0)),
            (218, (-1.0, 1.0, -9.0)),
            (219, (-1.0, 1.0, -9.0)),
            (220, (1.5, -1.0, -9.0)),
            (221, (1.5, -1.0, -9.0)),
            (222, (1.5, -1.0, -9.0)),
            (223, (-1.0, -1.0, -9.0)),
            (224, (-1.0, -1.0, -9.0)),
            (225, (-1.0, -1.0, -9.0)),
            (226, (1.5, 1.0, -9.0)),
            (227, (1.5, 1.0, -9.0)),
            (228, (1.5, 1.0, -9.0)),
            (229, (-1.0, 1.0, -9.0)),
            (230, (-1.0, 1.0, -9.0)),
            (231, (-1.0, 1.0, -9.0))
        ]



    
    members = [
            (0, 3),
            (1, 2),
            (0, 1),
            (2, 3),
            (0, 2),
            (1, 3),
            (4, 5),
            (6, 7),
            (4, 6),
            (5, 7),
            (1, 5),
            (0, 4),
            (3, 7),
            (2, 6),
            (5, 9),
            (4, 8),
            (7, 11),
            (6, 10),
            (1, 4),
            (0, 5),
            (1, 7),
            (3, 5),
            (3, 6),
            (2, 7),
            (0, 6),
            (2, 4),
            (5, 8),
            (4, 9),
            (5, 11),
            (7, 9),
            (7, 10),
            (6, 11),
            (6, 8),
            (4, 10),
            (8, 9),
            (10, 11),
            (8, 10),
            (9, 11),
            (12, 13),
            (14, 15),
            (12, 14),
            (13, 15),
            (16, 17),
            (18, 19),
            (16, 18),
            (17, 19),
            (9, 13),
            (8, 12),
            (11, 15),
            (10, 14),
            (13, 17),
            (12, 16),
            (15, 19),
            (16, 18),
            (17, 21),
            (16, 20),
            (19, 23),
            (18, 22),
            (8, 13),
            (9, 12),
            (9, 15),
            (11, 13),
            (11, 14),
            (10, 15),
            (10, 12),
            (8, 14),
            (12, 17),
            (13, 16),
            (13, 19),
            (15, 17),
            (15, 18),
            (14, 19),
            (14, 16),
            (12, 18),
            (16, 21),
            (17, 20),
            (17, 23),
            (19, 21),
            (19, 22),
            (18, 23),
            (18, 20),
            (16, 22),
            (20, 21),
            (22, 23),
            (20, 22),
            (21, 23),
            (21, 24),
            (20, 26),
            (21, 31),
            (23, 26),
            (23, 29),
            (22, 31),
            (22, 24),
            (20, 29),
            (21, 26),
            (20, 24),
            (23, 31),
            (22, 29),
            (21, 25),
            (20, 25),
            (21, 28),
            (23, 28),
            (23, 30),
            (22, 30),
            (22, 27),
            (20, 27),
            (24, 25),
            (25, 26),
            (24, 27),
            (27, 29),
            (29, 30),
            (30, 31),
            (26, 28),
            (28, 31),
            (32, 33),
            (33, 34),
            (32, 35),
            (35, 37),
            (37, 38),
            (38, 39),
            (34, 36),
            (36, 39),
            (24, 32),
            (26, 34),
            (29, 37),
            (31, 39),
            (32, 40),
            (34, 42),
            (37, 45),
            (39, 47),
            (24, 33),
            (25, 32),
            (25, 34),
            (26, 33),
            (27, 32),
            (24, 35),
            (26, 36),
            (28, 34),
            (28, 39),
            (31, 36),
            (31, 38),
            (30, 39),
            (30, 37),
            (29, 38),
            (29, 35),
            (27, 37),
            (32, 41),
            (33, 40),
            (33, 42),
            (34, 41),
            (34, 44),
            (36, 42),
            (36, 47),
            (39, 44),
            (39, 46),
            (38, 47),
            (38, 45),
            (37, 46),
            (37, 43),
            (35, 45),
            (35, 40),
            (32, 43),
            (25, 33),
            (27, 35),
            (30, 38),
            (28, 36),
            (33, 41),
            (35, 43),
            (38, 46),
            (36, 44),
            (40, 41),
            (41, 42),
            (40, 43),
            (42, 44),
            (43, 45),
            (45, 46),
            (46, 47),
            (44, 47),
            (48, 49),
            (49, 50),
            (48, 51),
            (50, 52),
            (51, 53),
            (53, 54),
            (54, 55),
            (52, 55),
            (42, 50),
            (40, 48),
            (45, 53),
            (47, 55),
            (48, 56),
            (50, 58),
            (53, 61),
            (55, 63),
            (40, 49),
            (41, 48),
            (41, 50),
            (42, 49),
            (42, 52),
            (44, 50),
            (44, 55),
            (47, 52),
            (47, 54),
            (46, 55),
            (46, 53),
            (45, 54),
            (45, 51),
            (43, 53),
            (43, 48),
            (40, 51),
            (48, 57),
            (49, 56),
            (49, 58),
            (50, 57),
            (50, 60),
            (52, 58),
            (52, 63),
            (55, 60),
            (55, 62),
            (54, 63),
            (54, 61),
            (53, 62),
            (53, 59),
            (51, 61),
            (51, 56),
            (48, 59),
            (41, 49),
            (43, 51),
            (44, 52),
            (46, 54),
            (49, 57),
            (51, 59),
            (52, 60),
            (54, 62),
            (56, 57),
            (57, 58),
            (56, 59),
            (58, 60),
            (59, 61),
            (60, 63),
            (61, 62),
            (62, 63),
            (64, 65),
            (65, 66),
            (64, 67),
            (66, 68),
            (67, 69),
            (68, 71),
            (69, 70),
            (70, 71),
            (72, 73),
            (73, 74),
            (72, 75),
            (74, 76),
            (75, 77),
            (76, 79),
            (77, 78),
            (78, 79),
            (56, 64),
            (58, 66),
            (61, 69),
            (63, 71),
            (64, 72),
            (66, 74),
            (69, 77),
            (71, 79),
            (72, 80),
            (74, 82),
            (77, 85),
            (79, 87),
            (56, 65),
            (57, 64),
            (57, 66),
            (58, 65),
            (58, 68),
            (60, 66),
            (60, 71),
            (63, 68),
            (63, 70),
            (62, 71),
            (62, 69),
            (61, 70),
            (61, 67),
            (59, 69),
            (59, 64),
            (56, 67),
            (64, 73),
            (65, 72),
            (65, 74),
            (66, 73),
            (66, 76),
            (68, 74),
            (68, 79),
            (71, 76),
            (71, 78),
            (70, 79),
            (70, 77),
            (69, 78),
            (69, 75),
            (67, 77),
            (67, 72),
            (64, 75),
            (72, 81),
            (73, 80),
            (73, 82),
            (74, 81),
            (74, 84),
            (76, 82),
            (76, 87),
            (79, 84),
            (79, 86),
            (78, 87),
            (78, 85),
            (77, 86),
            (77, 83),
            (75, 85),
            (75, 80),
            (72, 83),
            (57, 65),
            (59, 67),
            (60, 68),
            (62, 70),
            (65, 73),
            (67, 75),
            (68, 76),
            (70, 78),
            (73, 81),
            (75, 83),
            (76, 84),
            (78, 86),
            (80, 81),
            (81, 82),
            (80, 83),
            (82, 84),
            (83, 85),
            (84, 87),
            (85, 86),
            (86, 87),
            (80, 89),
            (82, 92),
            (85, 95),
            (87, 98),
            (81, 89),
            (81, 92),
            (84, 92),
            (84, 98),
            (83, 95),
            (83, 89),
            (86, 95),
            (86, 98),
            (82, 91),
            (80, 90),
            (82, 93),
            (87, 99),
            (80, 88),
            (85, 94),
            (85, 96),
            (87, 97),
            (81, 91),
            (81, 90),
            (83, 88),
            (83, 94),
            (84, 93),
            (84, 99),
            (86, 96),
            (86, 97),
            (89, 90),
            (91, 92),
            (88, 89),
            (92, 93),
            (94, 95),
            (95, 96),
            (97, 98),
            (98, 99),
            (101, 102),
            (103, 104),
            (100, 101),
            (104, 105),
            (106, 107),
            (107, 108),
            (109, 110),
            (110, 111),
            (90, 91),
            (88, 94),
            (93, 99),
            (96, 97),
            (102, 103),
            (100, 106),
            (105, 111),
            (108, 109),
            (89, 101),
            (92, 104),
            (95, 107),
            (98, 110),
            (101, 113),
            (104, 116),
            (107, 119),
            (110, 122),
            (92, 103),
            (91, 104),
            (89, 102),
            (90, 101),
            (88, 101),
            (89, 100),
            (92, 105),
            (93, 104),
            (99, 110),
            (98, 111),
            (98, 109),
            (97, 110),
            (96, 107),
            (95, 108),
            (95, 106),
            (94, 107),
            (101, 114),
            (102, 113),
            (101, 112),
            (100, 113),
            (103, 116),
            (104, 115),
            (104, 117),
            (105, 116),
            (111, 122),
            (110, 123),
            (110, 121),
            (109, 122),
            (108, 119),
            (107, 120),
            (107, 118),
            (106, 119),
            (90, 102),
            (91, 103),
            (96, 108),
            (97, 109),
            (88, 100),
            (94, 106),
            (93, 105),
            (99, 111),
            (102, 114),
            (103, 115),
            (108, 120),
            (109, 121),
            (100, 112),
            (106, 118),
            (105, 117),
            (111, 123),
            (90, 103),
            (91, 102),
            (96, 109),
            (97, 108),
            (88, 106),
            (94, 100),
            (93, 111),
            (99, 105),
            (102, 115),
            (103, 114),
            (108, 121),
            (109, 120),
            (105, 123),
            (111, 117),
            (100, 118),
            (106, 112),
            (112, 113),
            (113, 114),
            (115, 116),
            (116, 117),
            (118, 119),
            (119, 120),
            (121, 122),
            (122, 123),
            (124, 125),
            (125, 126),
            (127, 128),
            (128, 129),
            (130, 131),
            (131, 132),
            (133, 134),
            (134, 135),
            (136, 137),
            (137, 138),
            (139, 140),
            (140, 141),
            (142, 143),
            (143, 144),
            (145, 146),
            (146, 147),
            (114, 115),
            (120, 121),
            (112, 118),
            (117, 123),
            (126, 127),
            (132, 133),
            (124, 130),
            (129, 135),
            (138, 139),
            (144, 145),
            (136, 142),
            (141, 147),
            (113, 125),
            (116, 128),
            (119, 131),
            (122, 134),
            (125, 137),
            (128, 140),
            (131, 143),
            (134, 146),
            (137, 149),
            (140, 152),
            (143, 155),
            (146, 158),
            (112, 125),
            (113, 124),
            (113, 126),
            (114, 125),
            (115, 128),
            (116, 127),
            (116, 129),
            (117, 128),
            (118, 131),
            (119, 130),
            (119, 132),
            (120, 131),
            (121, 134),
            (122, 133),
            (122, 135),
            (123, 134),
            (124, 137),
            (125, 136),
            (125, 138),
            (126, 137),
            (127, 140),
            (128, 139),
            (128, 141),
            (129, 140),
            (130, 143),
            (131, 142),
            (131, 144),
            (132, 143),
            (133, 146),
            (134, 145),
            (134, 147),
            (135, 146),
            (136, 149),
            (137, 148),
            (137, 150),
            (138, 149),
            (139, 152),
            (140, 151),
            (140, 153),
            (141, 152),
            (142, 155),
            (143, 154),
            (143, 156),
            (144, 155),
            (145, 158),
            (146, 157),
            (146, 159),
            (147, 158),
            (114, 126),
            (115, 127),
            (120, 132),
            (121, 133),
            (112, 124),
            (118, 130),
            (117, 129),
            (123, 135),
            (126, 138),
            (127, 139),
            (132, 144),
            (133, 145),
            (124, 136),
            (130, 142),
            (129, 141),
            (135, 147),
            (138, 150),
            (139, 151),
            (144, 156),
            (145, 157),
            (136, 148),
            (142, 154),
            (141, 153),
            (147, 159),
            (114, 127),
            (115, 126),
            (120, 133),
            (121, 132),
            (112, 130),
            (118, 124),
            (117, 135),
            (123, 129),
            (126, 139),
            (127, 138),
            (132, 145),
            (133, 144),
            (124, 142),
            (130, 136),
            (129, 147),
            (135, 141),
            (138, 151),
            (139, 150),
            (144, 157),
            (145, 156),
            (136, 154),
            (142, 148),
            (141, 153),
            (147, 153),
            (148, 149),
            (149, 150),
            (151, 152),
            (152, 153),
            (154, 155),
            (155, 156),
            (157, 158),
            (158, 159),
            (160, 161),
            (161, 162),
            (163, 164),
            (164, 165),
            (166, 167),
            (167, 168),
            (169, 170),
            (170, 171),
            (172, 173),
            (173, 174),
            (175, 176),
            (176, 177),
            (178, 179),
            (179, 180),
            (181, 182),
            (182, 183),
            (150, 151),
            (156, 157),
            (148, 154),
            (153, 159),
            (162, 163),
            (168, 169),
            (160, 166),
            (165, 171),
            (174, 175),
            (180, 181),
            (172, 178),
            (177, 183),
            (149, 161),
            (152, 164),
            (155, 167),
            (158, 170),
            (161, 173),
            (164, 176),
            (167, 179),
            (170, 182),
            (173, 185),
            (176, 188),
            (179, 191),
            (182, 194),
            (148, 161),
            (149, 160),
            (149, 162),
            (150, 161),
            (151, 164),
            (152, 163),
            (152, 165),
            (153, 164),
            (154, 167),
            (155, 166),
            (155, 168),
            (156, 167),
            (157, 170),
            (158, 169),
            (158, 171),
            (159, 170),
            (160, 173),
            (161, 172),
            (161, 174),
            (162, 173),
            (163, 176),
            (164, 175),
            (164, 177),
            (165, 176),
            (166, 179),
            (167, 178),
            (167, 180),
            (168, 179),
            (169, 182),
            (170, 181),
            (170, 183),
            (171, 182),
            (172, 185),
            (173, 184),
            (173, 186),
            (174, 185),
            (175, 188),
            (176, 187),
            (176, 189),
            (177, 188),
            (178, 191),
            (179, 190),
            (179, 192),
            (180, 191),
            (181, 194),
            (182, 193),
            (182, 195),
            (183, 194),
            (150, 162),
            (151, 163),
            (156, 168),
            (157, 169),
            (148, 160),
            (154, 166),
            (153, 165),
            (159, 171),
            (162, 174),
            (163, 175),
            (168, 180),
            (169, 181),
            (160, 172),
            (166, 178),
            (165, 177),
            (171, 183),
            (174, 186),
            (175, 187),
            (180, 192),
            (181, 193),
            (172, 184),
            (178, 190),
            (177, 189),
            (183, 195),
            (150, 163),
            (151, 162),
            (156, 169),
            (157, 168),
            (148, 166),
            (154, 160),
            (153, 171),
            (159, 165),
            (162, 175),
            (163, 174),
            (168, 181),
            (169, 180),
            (160, 178),
            (166, 172),
            (165, 183),
            (171, 177),
            (174, 187),
            (175, 186),
            (180, 193),
            (180, 192),
            (172, 190),
            (178, 184),
            (177, 195),
            (183, 189),
            (184, 185),
            (185, 186),
            (187, 188),
            (188, 189),
            (190, 191),
            (191, 192),
            (193, 194),
            (194, 195),
            (196, 197),
            (197, 198),
            (199, 200),
            (200, 201),
            (202, 203),
            (203, 204),
            (205, 206),
            (206, 207),
            (208, 209),
            (209, 210),
            (211, 212),
            (212, 213),
            (214, 215),
            (215, 216),
            (217, 218),
            (218, 219),
            (186, 187),
            (192, 193),
            (184, 190),
            (189, 195),
            (198, 199),
            (204, 205),
            (196, 202),
            (201, 207),
            (210, 211),
            (216, 217),
            (208, 214),
            (213, 219),
            (185, 197),
            (188, 200),
            (191, 203),
            (194, 206),
            (197, 209),
            (200, 212),
            (203, 215),
            (206, 218),
            (209, 221),
            (212, 224),
            (215, 227),
            (218, 230),
            (184, 197),
            (185, 196),
            (185, 198),
            (186, 197),
            (187, 200),
            (188, 199),
            (188, 201),
            (189, 200),
            (190, 203),
            (191, 202),
            (191, 204),
            (192, 203),
            (193, 206),
            (194, 205),
            (194, 207),
            (195, 206),
            (196, 209),
            (197, 208),
            (197, 210),
            (198, 209),
            (199, 212),
            (200, 211),
            (200, 213),
            (201, 212),
            (202, 215),
            (203, 214),
            (203, 216),
            (204, 215),
            (205, 218),
            (206, 217),
            (206, 219),
            (207, 218),
            (208, 221),
            (209, 220),
            (209, 222),
            (210, 221),
            (211, 224),
            (212, 223),
            (212, 225),
            (213, 224),
            (214, 227),
            (215, 226),
            (215, 228),
            (216, 227),
            (217, 230),
            (218, 229),
            (218, 231),
            (219, 230),
            (186, 198),
            (187, 199),
            (192, 204),
            (193, 205),
            (184, 196),
            (190, 202),
            (189, 201),
            (195, 207),
            (198, 210),
            (199, 211),
            (204, 216),
            (205, 217),
            (196, 208),
            (202, 214),
            (201, 213),
            (207, 219),
            (210, 222),
            (211, 223),
            (216, 228),
            (217, 229),
            (208, 220),
            (214, 226),
            (213, 225),
            (219, 231),
            (186, 199),
            (187, 198),
            (192, 205),
            (193, 204),
            (184, 202),
            (190, 196),
        (189, 207),
        (195, 201),
        (198, 211),
        (199, 210),
        (204, 217),
        (205, 216),
        (196, 214),
        (202, 208),
        (201, 219),
        (207, 213),
        (210, 223),
        (211, 222),
        (216, 229),
        (217, 228),
        (208, 220),
        (214, 226),
        (213, 225),
        (219, 231),
        (220, 221),
        (221, 222),
        (223, 224),
        (224, 225),
        (226, 227),
        (227, 228),
        (229, 230),
        (230, 231),
        (222, 223),
        (228, 229),
        (220, 226),
        (225, 231),
        (221, 233),
        (224, 236),
        (227, 239),
        (230, 242),
        (222, 233),
        (220, 233),
        (223, 236),
        (225, 236),
        (226, 239),
        (228, 239),
        (229, 242),
        (231, 242),
        (221, 234),
        (221, 232),
        (224, 235),
        (224, 237),
        (227, 238),
        (227, 240),
        (230, 241),
        (230, 243),
        (222, 234),
        (223, 235),
        (228, 240),
        (229, 241),
        (220, 232),
        (226, 238),
        (225, 237),
        (231, 243),
        (222, 235),
        (223, 234),
        (228, 241),
        (229, 240),
        (220, 238),
        (226, 232),
        (225, 243),
        (231, 237)
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
        # memberType = MemberType(A[index].item(), 10000.0, 0.1)

        memberType = MemberType(A[index].item(), 10000.0, 0.1)

        if (E != None) & (Rho!=None):
            memberType = MemberType(A[index].item(), E[index].item(), Rho[index].item())
        elif (E != None) & (Rho==None):
            memberType = MemberType(A[index].item(), E[index].item(), 0.1)
        elif (E == None) & (Rho!=None):   
            memberType = MemberType(A[index].item(), 10000.0, Rho[index].item())

        truss.AddNewMember(jointID0, jointID1, memberType)
        index += 1

    # Do direct stiffness method:
    truss.Solve()

    # Dump all the structural analysis results into a .json file:
    # truss.DumpIntoJSON(TEST_OUTPUT_FILE)
    
    # Get result of structural analysis:
    displace, stress, resistance = truss.GetDisplacements(), truss.GetInternalStresses(), truss.GetResistances()
    return displace, stress, resistance, truss, truss.weight






##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

from .cont_to_disc import cont_to_disc

AISC_cross_sections = torch.tensor([
    0.111, 0.141, 0.196, 0.25, 0.307, 0.391, 0.442, 0.563, 0.602, 0.766, 0.785, 0.994, 1.0, 1.228, 1.266, 1.457,
    1.563, 1.620, 1.80, 1.990, 2.130, 2.380, 2.620, 2.63, 2.88, 2.93, 3.09, 3.13, 3.38, 3.47, 3.55, 3.63, 
    3.84, 3.87, 3.88, 4.18, 4.22, 4.49, 4.59, 4.80, 4.97, 5.12, 5.74, 7.22, 7.97, 8.53, 9.30, 10.85,
    11.50, 13.50, 13.90, 14.20, 15.50, 16.00, 16.90, 18.80, 19.90, 22.00, 22.90, 24.50, 26.50, 28.00, 30.00, 33.50
])

def Truss10D_Scaling(X):
    assert torch.is_tensor(X) and X.size(1) == 10, "Input must be an n-by-10 PyTorch tensor."
    X_scaled = X * (35-0.1) + 0.1
    
    return X_scaled

def Truss10D_MixedScaling(X):
    assert torch.is_tensor(X) and X.size(1) == 10, "Input must be an n-by-10 PyTorch tensor."

    D = torch.tensor([
        1.62, 1.80, 1.99, 2.13, 2.38, 2.62, 2.63, 2.88, 2.93, 3.09, 3.13, 3.38, 3.47, 3.55, 3.63, 
        3.84, 3.87, 3.88, 4.18, 4.22, 4.49, 4.59, 4.80, 4.97, 5.12, 5.74, 7.22, 7.97, 11.50, 13.50, 
        13.90, 14.20, 15.50, 16.00, 16.90, 18.80, 19.90, 22.00, 22.90, 26.50, 30.00, 33.50
    ])

    
    X_scaled = cont_to_disc(X, D)

    return X_scaled
    
    
def Truss25D_Scaling(X):
    assert torch.is_tensor(X) and (X.size(1) == 8 or X.size(1) == 25), "Input must be an n-by-8 or n-by-25 PyTorch tensor."
    X_scaled = X * (3.4-0.1) + 0.1
    
    return X_scaled   

def Truss25D_MixedScaling(X):
    assert torch.is_tensor(X) and (X.size(1) == 8 or X.size(1) == 25), "Input must be an n-by-8 or n-by-25 PyTorch tensor."

    D = torch.tensor([
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 
        1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4
    ]) 

    X_scaled = cont_to_disc(X, D)

    return X_scaled


def Truss72D_Scaling(X):
    assert torch.is_tensor(X) and (X.size(1) == 16 or X.size(1) == 72), "Input must be an n-by-16 or n-by-72 PyTorch tensor."
    X_scaled = X * (33.5 - 0.1) + 0.1
    
    return X_scaled    


def Truss72D_MixedScaling(X):
    assert torch.is_tensor(X) and (X.size(1) == 16 or X.size(1) == 72), "Input must be an n-by-16 or n-by-72 PyTorch tensor."
    X_scaled = cont_to_disc(X, AISC_cross_sections)

    return X_scaled



def Truss120D_Scaling(X):
    assert torch.is_tensor(X) and X.size(1) == 120, "Input must be an n-by-120 PyTorch tensor."
    X_scaled = X * (20-0.775) + 0.775
    
    return X_scaled  


def Truss200D_Scaling(X):
    assert torch.is_tensor(X) and (X.size(1) == 200 or X.size(1) == 29), "Input must be an n-by-200 or n-by-29 PyTorch tensor."
    X_scaled = X * (33.7-0.1) + 0.1
    
    return X_scaled

def Truss200D_MixedScaling(X):
    assert torch.is_tensor(X) and (X.size(1) == 200 or X.size(1) == 29), "Input must be an n-by-200 or n-by-29 PyTorch tensor."
    
    D = torch.tensor([
        0.100, 0.347, 0.440, 0.539, 0.954, 1.081, 1.174, 1.333, 1.488, 1.764, 2.142, 2.697, 
        2.800, 3.131, 3.565, 3.813, 4.805, 5.952, 6.572, 7.192, 8.525, 9.300, 10.850, 13.330, 
        14.290, 17.170, 19.180, 23.680, 28.080, 33.700
    ])

    X_scaled = cont_to_disc(X, D)

    return X_scaled

def Truss360D_Scaling(X):
    assert torch.is_tensor(X) and X.size(1) == 360, "Input must be an n-by-360 PyTorch tensor."
    X_scaled = torch.zeros(X.shape)
    X_scaled[:,:120] = X[:,:120] * (20-0.775) + 0.775
    X_scaled[:,120:240] = X[:,120:240] * (5e4-1e3) + 1e3
    X_scaled[:,240:360] = X[:,240:360] * (0.3-0.05) + 0.05
    
    return X_scaled  




def Truss10D(A): 
    assert torch.is_tensor(A) and A.size(1) == 10, "Input must be an n-by-10 PyTorch tensor."
    E = 1e7 * torch.ones(10)
    Rho = 0.1 * torch.ones(10)

    n = A.size(0)
    
    fx = torch.zeros(n,1)

    # 10 bar stress constraints, 4 displacement constraints
    gx = torch.zeros(n,14)

    for ii in range(n):
        
        displace, _, stress, _, _, weights = Truss10bar(A[ii,:], E, Rho)

        fx[ii,0] = -weights           # Negate for maximizing optimization

        for ss in range(10):
            gx[ii,ss] = abs(stress[ss]) - 25000
            
        gx[ii,10] = max(abs(displace[0][0]), abs(displace[0][1])) - 2
        gx[ii,11] = max(abs(displace[1][0]), abs(displace[1][1])) - 2
        gx[ii,12] = max(abs(displace[2][0]), abs(displace[2][1])) - 2
        gx[ii,13] = max(abs(displace[3][0]), abs(displace[3][1])) - 2

    return gx, fx



def Truss25D(A_): 
    assert torch.is_tensor(A_) and (A_.size(1) == 25 or A_.size(1) == 8), "Input must be an n-by-25 or n-by-8 PyTorch tensor."

    if A_.size(1) == 25:
        A = A_
    elif A_.size(1) == 8:
        # Bars in 8 groups because of symmetry
        # (1) A1, (2) A2–A5, (3) A6–A9, (4) A10–A11, (5) A12–A13, (6) A14–A17, (7) A18–A21 and (8) A22–A25.
        A = torch.zeros(A_.size(0), 25)
        A[:,0] = A_[:,0]
        A[:,1:5] = A_[:,1]
        A[:,5:9] = A_[:,2]
        A[:,9:11] = A_[:,3]
        A[:,11:13] = A_[:,4]
        A[:,13:17] = A_[:,5]
        A[:,17:21] = A_[:,6]
        A[:,21:25] = A_[:,7]

    E = 1e7 * torch.ones(25)
    Rho = 0.1 * torch.ones(25)


    n = A.size(0)
    
    fx = torch.zeros(n,1)

    # 25 bar stress constraints, 6 displacement constraints
    gx = torch.zeros(n,31)

    for ii in range(n):
        
        displace, _, stress, _, _, weights = Truss25bar(A[ii,:], E, Rho)

        fx[ii,0] = -weights           # Negate for maximizing optimization

        # Max stress less than 40ksi
        for ss in range(25):
            gx[ii,ss] = abs(stress[ss]) - 40000
        
        # Max displacement in x and y direction less than .35 inches
        for dd in range(6):
            # print(displace[dd])
            gx[ii,25+dd] = max(abs(displace[dd][0]), abs(displace[dd][1])) - 0.35


    return gx, fx




def Truss72D(A_, version="4_forces"): 
    assert torch.is_tensor(A_) and (A_.size(1) == 72 or A_.size(1) == 16), "Input must be an n-by-72 or n-by-16 PyTorch tensor."

    if A_.size(1) == 72:
        A = A_
    elif A_.size(1) == 16:
        # Bars in 16 groups because of symmetry
        # (1) A1–A4, (2) A5–A12, (3) A13–A16, (4) A17–A18, (5) A19–A22, (6) A23–A30, (7) A31–A34, (8) A35–A36,
        # (9) A37–A40, (10) A41–A48, (11) A49–A52, (12) A53–A54, (13) A55–A58, (14) A59–A66 (15), A67–A70, and (16) A71–A72.
        A = torch.zeros(A_.size(0), 72)
        A[:,0:4] = A_[:,0]
        A[:,4:12] = A_[:,1]
        A[:,12:16] = A_[:,2]
        A[:,16:18] = A_[:,3]
        A[:,18:22] = A_[:,4]
        A[:,22:30] = A_[:,5]
        A[:,30:34] = A_[:,6]
        A[:,34:36] = A_[:,7]
        A[:,36:40] = A_[:,8]
        A[:,40:48] = A_[:,9]
        A[:,48:52] = A_[:,10]
        A[:,52:54] = A_[:,11]
        A[:,54:58] = A_[:,12]
        A[:,58:66] = A_[:,13]
        A[:,66:70] = A_[:,14]
        A[:,70:72] = A_[:,15]

    E = 1e7 * torch.ones(72)
    Rho = 0.1 * torch.ones(72)

    n = A.size(0)
    
    fx = torch.zeros(n,1)

    # 72 bar stress constraints, 16 displacement constraints
    gx = torch.zeros(n, 88)

    for ii in range(n):
        
        displace, _, stress, _, _, weights = Truss72bar(A[ii,:], E, Rho, version)

        fx[ii,0] = -weights           # Negate for maximizing optimization

        # Max stress less than 25000
        for ss in range(72):
            gx[ii,ss] = abs(stress[ss]) - 25000
        
        # Max displacement in x and y direction less than .25 inches
        for dd in range(4, 20): # 16 free nodes
            gx[ii,72+dd-4] = max(abs(displace[dd][0]), abs(displace[dd][1])) - 0.25


    return gx, fx



def Truss120D(A): 

    assert torch.is_tensor(A) and A.size(1) == 120, "Input must be an n-by-120 PyTorch tensor."
    E = None
    Rho = None

    n = A.size(0)
    
    fx = torch.zeros(n,1)

    # 120 bar stress constraints, 1 displacement constraints
    gx = torch.zeros(n,121)

    for ii in range(n):
        # print(ii)
        # print(A[ii,:].shape)
        displace, _, stress, _, _, weights = Truss120bar(A[ii,:], None, None)

        if (E != None) & (Rho!=None):
            displace, _, stress, _, _, weights = Truss120bar(A[ii,:], E[ii,:], Rho[ii,:])
        elif (E != None) & (Rho==None):
            displace, _, stress, _, _, weights = Truss120bar(A[ii,:], E[ii,:], None)
        elif (E == None) & (Rho!=None):   
            displace, _, stress, _, _, weights = Truss120bar(A[ii,:], None, Rho[ii,:])

        fx[ii,0] = -weights           # Negate for maximizing optimization

        # Max stress less than 34800
        for ss in range(120):
            gx[ii,ss] = abs(stress[ss]) - 34800
        
        # Max displacement in x and y direction less than 
        MAX_DIST = 0
        for dd in range(len(displace)):
            if max(displace[dd]) > MAX_DIST:
                MAX_DIST = max(abs(displace[dd]))
        gx[ii,120] = MAX_DIST - 0.1969
        


    return gx, fx




def Truss200D(A_, version=1):
    assert torch.is_tensor(A_) and (A_.size(1) == 200 or A_.size(1) == 29), "Input must be an n-by-200 or n-by-29 PyTorch tensor."

    if A_.size(1) == 200:
        A = A_
    elif A_.size(1) == 29:
        # Bars in 29 groups because of symmetry
        # (1) A1-A4, (2) A5/8/11/14/17, (3) A19/20/21/22/23/24, (4) A18/25/56/63/94/101/132/139/170/177,
        # (5) A26/29/32/35/38, (6) A6/7/9/10/12/13/15/16/27/28/30/31/33/34/36/37, (7) A39/40/41/42, (8) A43/46/49/52/55,
        # (9) A57/58/59/60/61/62, (10) A64/67/70/73/76, (11) A44/45/47/48/50/51/53/54/65/66/68/69/71/72/74/75,
        # (12) A77/78/79/80, (13) A81/84/87/90/93, (14) A95/96/97/98/99/100, (15) A102/105/108/111/114, 
        # (16) A82/83/85/86/88/89/91/92/103/104/106/107/109/110/112/113, (17) A115/116/117/118, (18) A119/122/125/128/131,
        # (19) A133/134/135/136/137/138, (20) A140/143/146/149/152, (21) A120/121/123/124/126/127/129/130/141/142/144/145/147/148/150/151,
        # (22) A153/154/155/156, (23) A157/160/163/166/169, (24) A171/172/173/174/175/176, (25) A178/181/184/187/190,
        # (26) A158/159/161/162/164/165/167/168/179/180/182/183/185/186/188/189, (27) A191/192/193/194, (28) A195/197/198/200, (29) A196/199

        A = torch.zeros(A_.size(0), 200)
        A[:,0:4] = A_[:,0]
        A[:, [4, 7, 10, 13, 16]] = A_[:,1]
        A[:, [18, 19, 20, 21, 22, 23]] = A_[:,2]
        A[:, [17, 24, 55, 62, 93, 100, 131, 138, 169, 176]] = A_[:,3]
        A[:, [25, 28, 31, 34, 37]] = A_[:,4]
        A[:, [5, 6, 8, 9, 11, 12, 14, 15, 26, 27, 29, 30, 32, 33, 35, 36]] = A_[:,5]
        A[:, [38, 39, 40, 41]] = A_[:,6]
        A[:, [42, 45, 48, 51, 54]] = A_[:,7]
        A[:, [56, 57, 58, 59, 60, 61]] = A_[:,8]
        A[:, [63, 66, 69, 72, 75]] = A_[:,9]
        A[:, [43, 44, 46, 47, 49, 50, 52, 53, 64, 65, 67, 68, 70, 71, 73, 74]] = A_[:,10]
        A[:, [76, 77, 78, 79]] = A_[:,11]
        A[:, [80, 83, 86, 89, 92]] = A_[:,12]
        A[:, [94, 95, 96, 97, 98, 99]] = A_[:,13]
        A[:, [101, 104, 107, 110, 113]] = A_[:,14]
        A[:, [81, 82, 84, 85, 87, 88, 90, 91, 102, 103, 105, 106, 108, 109, 111, 112]] = A_[:,15]
        A[:, [114, 115, 116, 117]] = A_[:,16]
        A[:, [118, 121, 124, 127, 130]] = A_[:,17]
        A[:, [132, 133, 134, 135, 136, 137]] = A_[:,18]
        A[:, [139, 142, 145, 148, 151]] = A_[:,19]
        A[:, [119, 120, 122, 123, 125, 126, 128, 129, 140, 141, 143, 144, 146, 147, 149, 150]] = A_[:,20]
        A[:, [152, 153, 154, 155]] = A_[:,21]
        A[:, [156, 159, 162, 165, 168]] = A_[:,22]
        A[:, [170, 171, 172, 173, 174, 175]] = A_[:,23]
        A[:, [177, 180, 183, 186, 189]] = A_[:,24]
        A[:, [157, 158, 160, 161, 163, 164, 166, 167, 178, 179, 181, 182, 184, 185, 187, 188]] = A_[:,25]
        A[:, [190, 191, 192, 193]] = A_[:,26]
        A[:, [194, 196, 197, 199]] = A_[:,27]
        A[:, [195, 198]] = A_[:,28]

    
    E = 3e4 * torch.ones(200)
    Rho = 0.283 * torch.ones(200)

    n = A.size(0)

    fx = torch.zeros(n,1)

    # 200 bar stress constraints
    gx = torch.zeros(n, 200)

    for ii in range(n):
        
        displace, _, stress, _, _, weights = Truss200bar(A[ii,:], E, Rho, version)

        fx[ii,0] = -weights           # Negate for maximizing optimization

        # Max stress less than 10000
        for ss in range(200):
            if ss in stress:
                gx[ii,ss] = abs(stress[ss]) - 10000
            else:
                gx[ii,ss] = -10000
        

    return gx, fx






    







def Truss360D(X): 

    assert torch.is_tensor(X) and X.size(1) == 360, "Input must be an n-by-360 PyTorch tensor."
    A = X[:,:120]
    E = X[:,120:240]
    Rho = X[:,240:360]

    n = A.size(0)
    
    fx = torch.zeros(n,1)

    # 120 bar stress constraints, 1 displacement constraints
    gx = torch.zeros(n,121)

    for ii in range(n):
        # print(ii)
        # print(A[ii,:].shape)
        displace, _, stress, _, _, weights = Truss120bar(A[ii,:], E[ii,:], Rho[ii,:])

        # if (E != None) & (Rho!=None):
        #     displace, _, stress, _, _, weights = Truss120bar(A[ii,:], E[ii,:], Rho[ii,:])
        # elif (E != None) & (Rho==None):
        #     displace, _, stress, _, _, weights = Truss120bar(A[ii,:], E[ii,:], None)
        # elif (E == None) & (Rho!=None):   
        #     displace, _, stress, _, _, weights = Truss120bar(A[ii,:], None, Rho[ii,:])

        fx[ii,0] = -weights           # Negate for maximizing optimization

        # Max stress less than 34800
        for ss in range(120):
            gx[ii,ss] = abs(stress[ss]) - 34800
            # gx[ii,ss] = abs(stress[ss]) - 34800
        
        # Max displacement in x and y direction less than 
        MAX_DIST = 0
        for dd in range(len(displace)):
            if max(displace[dd]) > MAX_DIST:
                MAX_DIST = max(abs(displace[dd]))
        gx[ii,120] = MAX_DIST - 0.1969
        


    return gx, fx




































