import numpy as np


def get_thetas(curve_points):
    """广义极角，theta = 2*pi*s/l l为边界的长度，s为边界上的点到起点的距离"""
    # l = ∑(√((x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2))
    # s 为边界上的点到起点的距离
    s = [0]
    for i in range(1, len(curve_points), 1):
        s.append(s[i-1]+np.sqrt((curve_points[i][0]-curve_points[i-1]
                 [0])**2+(curve_points[i][1]-curve_points[i-1][1])**2))
    s.append(s[-1]+np.sqrt((curve_points[0][0]-curve_points[-1][0])
             ** 2+(curve_points[0][1]-curve_points[-1][1])**2))
    l = s[-1]
    # print(l)
    s = np.array(s)
    thetas = (s / l) * 2 * np.pi
    return thetas


def get_a_ns(thetas, curve_points, n):
    """求解傅立叶级数的系数 a_n """
    a_ns0 = []
    a_ns1 = []
    for i in range(-n, n+1, 1):
        if i == 0:
            r_0 = np.array([0.0, 0.0])
            for j in range(0, len(curve_points), 1):
                r_j = curve_points[j]
                theta_j = thetas[j]
                if j == len(curve_points)-1:
                    theta_j_1 = 2*np.pi
                    r_j_1 = curve_points[0]
                else:
                    theta_j_1 = thetas[j+1]
                    r_j_1 = curve_points[j+1]
                r_0 += (r_j+r_j_1)/2 * (theta_j_1-theta_j)
            r_0 /= (2*np.pi)
            a_ns0.append(0)
            a_ns1.append(0)
        # a_n = 1/2pi * ∫(f(t) * e^(nj)dt)
        else:
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
    return [a_ns0, a_ns1], r_0


def F(r_0, theta, a_ns, n):
    # F(θ) = ∑(a_n * e^(inθ))
    r = np.array([0.0, 0.0])
    for i in range(-n, n+1, 1):
        r += a_ns[0][i+n] * np.array([np.cos(i*theta), np.sin(i*theta)]) + \
            a_ns[1][i+n] * np.array([np.sin(i*theta), np.cos(i*theta)])
    return r_0+r
