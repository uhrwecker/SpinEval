import numpy as np
from scipy.optimize import fsolve


def exc(e, s, a, dif=1e-5):
    kappa_1 = 2 * (3 - 2 * e ** 2) / e ** 3 * np.arcsin(e)
    kappa_2 = 6 / e ** 2 * np.sqrt(1 - e ** 2)

    factor = 3 / np.pi

    return np.abs(s) - np.sqrt(dif * a) * np.sqrt(factor * (kappa_1 - kappa_2))


def calculate_polar_semi_axis(s, a, dif=1e-5):
    potenz = 1
    new_a = np.abs(a)
    while new_a < 1:
        new_a *= 10
        potenz *= 10
    dif = 1.3 / potenz
    print(dif)
    def func(e):
        kappa_1 = 2 * (3 - 2 * e**2) / e**3 * np.arcsin(e)
        kappa_2 = 6 / e**2 * np.sqrt(1 - e**2)

        factor = 3 / np.pi

        return np.abs(s) - np.sqrt(dif * a) * np.sqrt(factor * (kappa_1 - kappa_2))

    e = fsolve(func, 0.8)
    if len(e) > 0:
        c = a * np.sqrt(1 - e**2)
    else:
        raise ValueError('Could not calculate the polar semi-major axis c.')

    return c


import matplotlib.pyplot as pl
s = -0.002
a = 0.005
dif = 1.3e-3

pl.figure(figsize=(7, 5))
a = np.linspace(0.00005, 0.5, num=100000)
pl.plot(a, a/4.3, label=r'$(\mu \, / \, M)_{min}$ for $|s| = s_{max}$')
pl.xscale('log')
pl.yscale('log')
pl.xlabel('a / M')
pl.ylabel(r'$(\mu \, / \, M)_{min}$')
pl.xlim(a[0], a[-1])
pl.legend()
pl.grid()
pl.show()

print(calculate_polar_semi_axis(s, a, dif))

