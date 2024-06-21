import numpy as np


def get_thetas(curve_points):
    """广义极角，theta = 2*pi*s/l l为边界的长度，s为边界上的点到起点的距离"""
    # l = ∑(√((x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2))
    # s 为边界上的点到起点的距离

    curve_points = np.append(curve_points, [curve_points[0]], axis=0)
    x = curve_points[:, 0]
    y = curve_points[:, 1]

    ds = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    s = np.cumsum(ds)

    l = s[-1]
    thetas = (s / l) * 2 * np.pi
    return thetas


def get_a_ns(thetas, curve_points, n):
    """求解傅立叶级数的系数 a_n """
    a_ns0 = []
    a_ns1 = []
    for i in range(-n, n+1, 1):
        a_n0 = 0
        a_n1 = 0
        for j in range(0, len(curve_points), 1):
            r_j = curve_points[j]
            theta_j = thetas[j]
            if j == len(curve_points)-1:
                theta_j_1 = 2*np.pi
                r_j_1 = curve_points[0]
            else:
                theta_j_1 = thetas[j+1]
                r_j_1 = curve_points[j+1]
            rc_bar0 = (np.dot(r_j, [np.cos(i*theta_j), np.sin(i*theta_j)]) +
                       np.dot(r_j_1, [np.cos(i*theta_j_1), np.sin(i*theta_j_1)]))/2
            rc_bar1 = (np.dot(r_j, [np.sin(i*theta_j), np.cos(i*theta_j)]) +
                       np.dot(r_j_1, [np.sin(i*theta_j_1), np.cos(i*theta_j_1)]))/2
            a_n0 += rc_bar0 * (theta_j_1-theta_j)
            a_n1 += rc_bar1 * (theta_j_1-theta_j)
        a_n0 /= (2*np.pi)
        a_n1 /= (2*np.pi)
        a_ns0.append(a_n0)
        a_ns1.append(a_n1)
    # return [a_ns0, a_ns1], r_0
    return [a_ns0, a_ns1]


def F(theta, a_ns, n):
    # F(θ) = ∑(a_n * e^(inθ))
    r = np.array([0.0, 0.0])
    acun = int((len(a_ns[0]) - 1) / 2)
    for i in range(acun-n, acun+n+1, 1):
        r += a_ns[0][i] * np.array([np.cos((i-acun)*theta), np.sin((i-acun)*theta)]) + \
            a_ns[1][i] * np.array([np.sin((i-acun)*theta),
                                  np.cos((i-acun)*theta)])
    return r
