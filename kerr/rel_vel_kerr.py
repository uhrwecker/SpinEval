import numpy as np
import matplotlib.pyplot as pl


def _calculate_velocity(position, a, s):
    chi_v = _chi_v(position, a, s)

    chi_u = (position - s ** 2 / position ** 2) / \
            (position + 2 * s ** 2 / position ** 2) * chi_v

    v = 1 - (1 - chi_v ** 2) * (1 - chi_u ** 2) / (1 - chi_u * chi_v) ** 2

    return v


def _chi_v(position, a, s):
    """Claculate Chi_v in particular. It is the same as the orbit velocity."""
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

    return v


pl.figure(figsize=(13, 5))

s = np.linspace(-0.002, 0.002, num=10000)
s = np.float32(s)

v0 = _calculate_velocity(8.0, 0.0, s)
v01 = _calculate_velocity(7.82691869, 0.1, s)
v05 = _calculate_velocity(7.05207426, 0.5, s)
v09 = _calculate_velocity(6.06747494, 0.9, s)
v_01 = _calculate_velocity(8.16655156, -0.1, s)
v_05 = _calculate_velocity(8.77840627, -0.5, s)
v_09 = _calculate_velocity(9.32167376, -0.9, s)

r = np.linspace(1, 10, num=100000)
v00 = _calculate_velocity(r, 1, 1)

pl.plot(r, v00)
pl.ylim(0, 0.1)

#pl.plot(s, v_09, label='a = -0.9, r = 9.32167376')
#pl.plot(s, v_05, label='a = -0.5, r = 8.77840627')
#pl.plot(s, v_01, label='a = -0.1, r = 8.16655156')
#pl.plot(s, v0, label='a =  0.0, r = 8.0')
#pl.plot(s, v01, label='a =  0.1, r = 7.82691869')
#pl.plot(s, v05, label='a =  0.5, r = 7.05207426')
#pl.plot(s, v09, label='a =  0.9, r = 6.06747494')
#print(v0)

#pl.xlim(s[0], s[-1])
#pl.ylim(-1e-30, 1e-30)
#pl.xlabel('s')
#pl.ylabel(r'orbital velocity v')
#pl.grid()
#pl.legend()

pl.show()