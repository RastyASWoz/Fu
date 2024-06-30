import numpy as np
from scipy import integrate


def get_thetas(curve_points):
    """广义极角，theta = 2*pi*s/l l为边界的长度，s为边界上的点到起点的距离"""
    # l = ∑(√((x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2))
    # s 为边界上的点到起点的距离

    x = curve_points[:, 0]
    y = curve_points[:, 1]

    ds = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    s = np.insert(np.cumsum(ds), 0, 0, axis=0)

    l = s[-1]
    thetas = (s / l) * 2 * np.pi
    return thetas


def get_a_ns(thetas, r, n):
    """求解傅立叶级数的系数 a_n """

    def dot_1(a, b):
        return np.multiply(a, b).sum(axis=1)

    a_ns0 = [
        integrate.trapezoid(
            dot_1(r, np.transpose([np.cos(i*thetas), np.sin(i*thetas)])),
            thetas,
        ) / (2*np.pi)
        for i in range(-n, n+1, 1)
    ]

    a_ns1 = [
        integrate.trapezoid(
            dot_1(r, np.transpose([np.sin(i*thetas), np.cos(i*thetas)])),
            thetas,
        ) / (2*np.pi)
        for i in range(-n, n+1, 1)
    ]

    return np.array([a_ns0, a_ns1])


def F(theta, a_ns, n):
    # F(θ) = ∑(a_n * e^(inθ))
    acun = (a_ns.shape[1] - 1) // 2
    coef = np.arange(-n, n+1)

    a0, a1 = a_ns[:, acun+coef]
    a0 = np.expand_dims(a0, axis=1)
    a1 = np.expand_dims(a1, axis=1)

    def outer(a, b):
        """Like np.outer but return 1-d array when b is a number"""
        return np.tensordot(a, b, ((), ()))

    vect = [np.cos(outer(coef, theta)), np.sin(outer(coef, theta))]
    vect1 = np.transpose(vect)
    vect2 = np.transpose(vect[::-1])
    # axis -3（如有）: 采样点数
    # axis -2: 级数阶数 2*n+1
    # axis -1: x 或 y 坐标
    r = (a0*vect1 + a1*vect2).sum(axis=-2)
    return r
