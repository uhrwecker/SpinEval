import numpy as np
from scipy.optimize import fsolve

def E(r, a):
    u = 1 / r
    return (1 - 2 * u - a * np.sqrt(u ** 3)) / np.sqrt(1 - 3 * u - 2 * a * np.sqrt(u ** 3))

def L(r, a):
    u = 1 / r
    return - np.sign(a+0.00001) * (1 + a ** 2 * u ** 2 + 2 * a * np.sqrt(u ** 3)) / np.sqrt(u * (1 - 3 * u - 2 * a * np.sqrt(u ** 3)))

def dphi(r, a):
    l = L(r, a)
    e = E(r, a)
    T = e * (r ** 2 + a ** 2) - l * a
    delta = r ** 2 - 2 * r + a ** 2
    return - (a * e - l) + a * T / delta

def dt(r, a):
    l = L(r, a)
    e = E(r, a)
    T = e * (r ** 2 + a ** 2) - l * a
    delta = r ** 2 - 2 * r + a ** 2
    return - a ** 2 * e + a * l + (r**2 + a ** 2) * T / delta

def OoO(r, a):
    return dt(r, a) / dphi(r, a)

def omega(r, a):
    return 1 / (r ** 1.5 + a)


def chi_v(r, a):
    f1 = - 2 * r * a
    root = 4 * r ** 3
    denom = 2 * np.sqrt(r**2 - 2 * r + a ** 2) * r
    return (f1 + np.sqrt(root)) / denom


def v0(r, a):
    return 1 / np.sqrt(1 - chi_v(r, a) ** 2)


def v3(r, a):
    return chi_v(r, a) * v0(r, a)


def O(r, a):
    delta = r ** 2 - 2 * r + a ** 2
    A = (r ** 2 + a ** 2) ** 2 - delta * a ** 2 * np.sin(np.pi / 2)
    sigma = r ** 2 + a ** 2 * np.cos(np.pi / 2) ** 2

    e_nu = np.sqrt((sigma * delta) / A)
    e_psi = np.sqrt((A * np.sin(np.pi / 2) ** 2) / sigma)
    omega = 2 * a * r / A

    v00, v33 = _calculate_velocity(r, a, 0.)

    return v33 / v00 #e_psi * v33 / (e_nu * v00 - omega * e_psi * v33)


def _calculate_velocity(position, a, s):
    root = 4 * position ** 3 + \
           12 * a * position * s + \
           13 * s ** 2 + \
           6 * a * s ** 3 / position ** 2 - \
           8 * s ** 4 / position ** 3 + \
           9 * a ** 2 * s ** 4 / position ** 4
    denominator = 2 * np.sqrt(position ** 2 - 2 * position + a ** 2) * (
            position - s ** 2 / position ** 2)

    v = (- 2 * position * a - 3 * s - a * s ** 2 / position ** 2
         + np.sqrt(root)) / denominator

    #print(v)

    return 1 / np.sqrt(1 - v**2), v / np.sqrt(1 - v**2)

import matplotlib.pyplot as pl

r = np.linspace(1, 15, num=10000)

def func(r, a):
    return O(r, a) * 123.12478369553894 / (2 * np.pi) - r

print(2 * np.pi * 8 / O(8., 0.))
#pl.plot(r, 1 / O(r, 0.))
#pl.plot(r, 1 / O(r, 0.99))
#pl.plot(r, 1 / O(r, -0.99))

a0 = fsolve(func, 8.0, (0.,))
a01 = fsolve(func, 8.0, (0.1,))
a05 = fsolve(func, 8.0, (0.5,))
a09 = fsolve(func, 8.0, (0.9,))
a_01 = fsolve(func, 8.0, (-0.1,))
a_05 = fsolve(func, 8.0, (-0.5,))
a_09 = fsolve(func, 8.0, (-0.9,))
print(a0)
print(a01, a05, a09)
print(a_01, a_05, a_09)
#pl.axhline(2)

#pl.show()