import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

f = None
D = None
h = None


def ybus(linedata, f):
    linedata[:, 3] = linedata[:, 3] * f

    fb = linedata[:, 0].astype(int)
    tb = linedata[:, 1].astype(int)
    r = linedata[:, 2]
    x = linedata[:, 3]
    b = linedata[:, 4]
    a = linedata[:, 5]

    z = r + 1j * x
    y = 1 / z
    b = 1j * b

    nb = max(max(fb), max(tb))
    nl = len(fb)
    Y = np.zeros((nb, nb), dtype=complex)

    for k in range(nl):
        Y[fb[k] - 1, tb[k] - 1] -= y[k] / a[k]
        Y[tb[k] - 1, fb[k] - 1] = Y[fb[k] - 1, tb[k] - 1]

    for m in range(nb):
        for n in range(nl):
            if fb[n] - 1 == m:
                Y[m, m] += y[n] / (a[n] ** 2) + b[n]
            elif tb[n] - 1 == m:
                Y[m, m] += y[n] + b[n]

    return Y


def OBJ11(x, n):
    a, b, c, e, f, l = x
    Zmax = 99.9999
    P = 100

    def fhd(z):
        term1 = np.arccos(
            (a**2 + (l - z) ** 2 + e**2 - b**2) / (2 * a * np.sqrt((l - z) ** 2 + e**2))
        )
        term2 = np.arccos(
            (b**2 + (l - z) ** 2 + e**2 - a**2) / (2 * b * np.sqrt((l - z) ** 2 + e**2))
        )
        term3 = np.arctan(e / (l - z))
        numerator = P * b * np.sin(term1 + term2)
        denominator = 2 * c * np.cos(term1 + term3)
        return numerator / denominator if n == 1 else -numerator / denominator

    result = minimize_scalar(fhd, bounds=(0, Zmax), method="bounded")
    return result.fun


def function_fitness(section):
    # E = 6.98e10
    A = section
    rho = 2770

    gcoord = np.array(
        [[18.288, 18.288, 9.144, 9.144, 0, 0], [9.144, 0, 9.144, 0, 9.144, 0]]
    )

    element = np.array([[3, 1, 4, 2, 3, 1, 4, 3, 2, 1], [5, 3, 6, 4, 4, 2, 5, 6, 3, 4]])

    Weight = 0
    for i in range(element.shape[1]):
        nd = element[:, i] - 1
        x = gcoord[0, nd]
        y = gcoord[1, nd]

        le = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
        Weight += rho * le * A[i]

    return Weight


def ConsBar10(x):
    type_ = "2D"
    E = 6.98e10
    A = x
    rho = 2770

    gcoord = np.array(
        [[18.288, 18.288, 9.144, 9.144, 0, 0], [9.144, 0, 9.144, 0, 9.144, 0]]
    )

    element = np.array([[3, 1, 4, 2, 3, 1, 4, 3, 2, 1], [5, 3, 6, 4, 4, 2, 5, 6, 3, 4]])

    # nel = element.shape[1]
    nnode = gcoord.shape[1]
    ndof = 2
    sdof = nnode * ndof

    K, M = Cal_K_and_M(type_, gcoord, element, A, rho, E)

    addedMass = 454
    for idof in range(sdof):
        M[idof, idof] += addedMass

    bcdof = np.array([(5 - 1) * 2, (5 - 1) * 2 + 1, (6 - 1) * 2, (6 - 1) * 2 + 1])

    omega_2 = eigens(K, M, bcdof)
    f = np.sqrt(omega_2) / (2 * np.pi)

    c1 = 7 / f[0] - 1
    c2 = 15 / f[1] - 1
    c3 = 20 / f[2] - 1

    c = np.array([c1, c2, c3])
    ceq = np.array([])

    return c, ceq


def Cal_K_and_M(type_, gcoord, element, A, rho, E):
    nel = element.shape[1]
    nnode = gcoord.shape[1]
    ndof = 2
    sdof = nnode * ndof
    K = np.zeros((sdof, sdof))
    M = np.zeros((sdof, sdof))

    for iel in range(nel):
        nd = element[:, iel] - 1
        x = gcoord[0, nd]
        y = gcoord[1, nd]

        le = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)

        l_ij = (x[1] - x[0]) / le
        m_ij = (y[1] - y[0]) / le

        Te = np.array([[l_ij, m_ij, 0, 0], [0, 0, l_ij, m_ij]])

        ke = (A[iel] * E / le) * np.array([[1, -1], [-1, 1]])
        ke = Te.T @ ke @ Te

        me = (rho * le * A[iel] / 6) * np.array(
            [[2, 0, 1, 0], [0, 2, 0, 1], [1, 0, 2, 0], [0, 1, 0, 2]]
        )

        index = np.array([2 * nd[0], 2 * nd[0] + 1, 2 * nd[1], 2 * nd[1] + 1])

        K[np.ix_(index, index)] += ke
        M[np.ix_(index, index)] += me

    return K, M


def eigens(K, M, b):
    nd = K.shape[0]
    fdof = np.arange(nd)
    pdof = b.flatten()
    fdof = np.delete(fdof, pdof)

    K_ff = K[np.ix_(fdof, fdof)]
    M_ff = M[np.ix_(fdof, fdof)]

    eigenvalues, _ = np.linalg.eig(np.linalg.inv(M_ff) @ K_ff)
    eigenvalues = np.sort(eigenvalues)

    return eigenvalues


def lk():
    E = 206000000.0
    nu = 0.3
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    KE = (
        E
        / (1 - nu**2)
        * np.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )
    return KE


def FE(nelx, nely, x, penal):
    KE = lk()
    ndof = 2 * (nelx + 1) * (nely + 1)
    K = lil_matrix((ndof, ndof))
    F = np.zeros((ndof, 1))
    U = np.zeros((ndof, 1))

    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof = np.array(
                [
                    2 * n1,
                    2 * n1 + 1,
                    2 * n2,
                    2 * n2 + 1,
                    2 * n2 + 2,
                    2 * n2 + 3,
                    2 * n1 + 2,
                    2 * n1 + 3,
                ]
            )
            K[np.ix_(edof, edof)] += x[ely, elx] ** penal * KE

    F[2 * (nely + 1) * (nelx + 1) - 1, 0] = -10000

    fixeddofs = np.arange(0, 2 * (nely + 1))
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    U[freedofs, 0] = spsolve(K[freedofs, :][:, freedofs], F[freedofs, 0])
    U[fixeddofs, 0] = 0

    return U


def check(nelx, nely, rmin, x, dc):
    dcn = np.zeros((nely, nelx))
    R = int(math.floor(rmin))
    for i in range(nelx):
        for j in range(nely):
            summation = 0.0

            for k in range(max(i - R, 0), min(i + R + 1, nelx)):
                for l in range(max(j - R, 0), min(j + R + 1, nely)):
                    fac = rmin - math.sqrt((i - k) ** 2 + (j - l) ** 2)
                    fac = max(0.0, fac)
                    summation += fac
                    dcn[j, i] += fac * x[l, k] * dc[l, k]
            dcn[j, i] = dcn[j, i] / (x[j, i] * summation)
    return dcn


thetaVeldefijMatrix = None
turbineMoved = None


def restrict(val, max_val):
    return min(val, max_val)


def downstream_wind_turbine_is_affected(
    coordinate, upstream, downstream, theta, kappa, R
):
    Tijx = coordinate[2 * downstream] - coordinate[2 * upstream]
    Tijy = coordinate[2 * downstream + 1] - coordinate[2 * upstream + 1]

    theta_rad = math.radians(theta)
    dij = math.cos(theta_rad) * Tijx + math.sin(theta_rad) * Tijy

    inner = (Tijx**2 + Tijy**2) - dij**2
    inner = max(inner, 0)
    lij = math.sqrt(inner)

    l = dij * kappa + R

    affected = (upstream != downstream) and (l > (lij - R)) and (dij > 0)
    return affected, dij


def eva_func_deficit(interval_dir_num, N, coordinate, theta, a, kappa, R):
    global thetaVeldefijMatrix
    thetaVeldefijMatrix = np.zeros((N, N, interval_dir_num))
    vel_def = np.zeros(N)

    idx = interval_dir_num - 1
    for i in range(N):
        vel_def_i = 0
        for j in range(N):
            affected, dij = downstream_wind_turbine_is_affected(
                coordinate, j, i, theta, kappa, R
            )
            if affected:
                d = a / (1 + kappa * dij / R) ** 2

                thetaVeldefijMatrix[i, j, idx] = d
                vel_def_i += d**2
            else:
                thetaVeldefijMatrix[i, j, idx] = 0

        vel_def[i] = math.sqrt(vel_def_i)
    return vel_def


def eva_func_deficit_caching(interval_dir_num, N, coordinate, theta, a, kappa, R):
    global thetaVeldefijMatrix, turbineMoved
    vel_def = np.zeros(N)
    idx = interval_dir_num - 1

    movedTurbine = None
    for i in range(N):
        if turbineMoved[i] == 1:
            movedTurbine = i
    if movedTurbine is None:
        movedTurbine = 0

    for i in range(N):
        vel_def_i = 0
        if i != movedTurbine:
            affected, dij = downstream_wind_turbine_is_affected(
                coordinate, movedTurbine, i, theta, kappa, R
            )
            if affected:
                d = a / (1 + kappa * dij / R) ** 2
                d = restrict(d, 1)
            else:
                d = 0

            vel_def_i = (
                np.sum(thetaVeldefijMatrix[i, :, idx] ** 2)
                - (thetaVeldefijMatrix[i, movedTurbine, idx]) ** 2
                + d**2
            )
            thetaVeldefijMatrix[i, movedTurbine, idx] = d
        else:
            for j in range(N):
                affected, dij = downstream_wind_turbine_is_affected(
                    coordinate, j, i, theta, kappa, R
                )
                if affected:
                    d = a / (1 + kappa * dij / R) ** 2
                    d = restrict(d, 1)
                else:
                    d = 0
                vel_def_i += d**2
                thetaVeldefijMatrix[i, j, idx] = d
        vel_def_i = restrict(vel_def_i, 1)
        vel_def[i] = math.sqrt(vel_def_i)
    return vel_def


def eva_power(
    interval_dir_num,
    interval_dir,
    N,
    coordinate,
    a,
    kappa,
    R,
    k_val,
    c_val,
    cut_in_speed,
    rated_speed,
    cut_out_speed,
    evaluate_method,
):
    if evaluate_method == "caching":
        vel_def = eva_func_deficit_caching(
            interval_dir_num, N, coordinate, interval_dir, a, kappa, R
        )
    else:
        vel_def = eva_func_deficit(
            interval_dir_num, N, coordinate, interval_dir, a, kappa, R
        )

    interval_c = c_val * (1 - vel_def)

    n_ws = int((rated_speed - cut_in_speed) / 0.3)
    power_eva = np.zeros(N)

    for i in range(N):
        for j in range(1, n_ws + 1):
            v_j_1 = cut_in_speed + (j - 1) * 0.3
            v_j = cut_in_speed + j * 0.3
            term1 = (
                1500
                * math.exp(((v_j_1 + v_j) / 2) - 7.5)
                / (5 + math.exp(((v_j_1 + v_j) / 2) - 7.5))
            )
            term2 = math.exp(-((v_j_1 / interval_c[i]) ** k_val)) - math.exp(
                -((v_j / interval_c[i]) ** k_val)
            )
            power_eva[i] += term1 * term2
        power_eva[i] += 1500 * (
            math.exp(-((rated_speed / interval_c[i]) ** k_val))
            - math.exp(-((cut_out_speed / interval_c[i]) ** k_val))
        )
    return power_eva


def Fitness(
    interval_num,
    interval,
    fre,
    N,
    coordinate,
    a,
    kappa,
    R,
    k_array,
    c_array,
    cut_in_speed,
    rated_speed,
    cut_out_speed,
    evaluate_method,
):
    all_power = 0
    for i in range(interval_num):
        interval_dir_num = i + 1
        interval_dir = (i + 0.5) * interval

        power_eva = eva_power(
            interval_dir_num,
            interval_dir,
            N,
            coordinate,
            a,
            kappa,
            R,
            k_array[i],
            c_array[i],
            cut_in_speed,
            rated_speed,
            cut_out_speed,
            evaluate_method,
        )
        all_power += fre[i] * np.sum(power_eva)
    return all_power
