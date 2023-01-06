import numpy as np


def relative_velocity(s, position):
    chi_v = orbit_velocity(s, position)

    chi_u = (position - s ** 2 / position ** 2) / \
            (position + 2 * s ** 2 / position ** 2) * chi_v

    v = 1 - (1 - chi_v ** 2) * (1 - chi_u ** 2) / (1 - chi_u * chi_v) ** 2

    return v#np.sqrt(v)


def orbit_velocity(s, position):
    """Claculate Chi_v in particular. It is the same as the orbit velocity."""
    root = 4 * position ** 3 + 13 * s ** 2 - 8 * s ** 4 / position ** 3
    denominator = 2 * np.sqrt(position ** 2 - 2 * position) * (
            position - s ** 2 / position ** 2)

    v = (-3 * s + np.sqrt(root)) / denominator

    return v


def retrieve_north_pole(fp):
    from os import listdir
    from os.path import isfile, join
    import json

    all_files = [f for f in listdir(fp) if isfile(join(fp, f))]
    smallest = 100
    ab = (0, 0)

    for file in all_files:
        if not file.endswith('.json'):
            continue
        with open(fp+file, 'r') as ff:
            data = json.load(ff)
        theta = data['EMITTER']['Theta']

        if np.abs(theta) < smallest:
            smallest = np.abs(theta)
            ab = (data['OBSERVER']['alpha'], data['OBSERVER']['beta'])

    return smallest, ab


def retrieve_eq(fp):
    from os import listdir
    from os.path import isfile, join
    import json

    all_files = [f for f in listdir(fp) if isfile(join(fp, f))]
    equator = []
    nearly = (np.pi/2 - 0.02, np.pi/2 + 0.02)

    for file in all_files:
        if not file.endswith('.json'):
            continue
        with open(fp+file, 'r') as ff:
            data = json.load(ff)
        theta = data['EMITTER']['Theta']

        if nearly[0] < theta < nearly[1]:
            equator.append((data['OBSERVER']['alpha'], data['OBSERVER']['beta']))

    return equator


def g(p0, p1, p3, orbit_vel, gamma_orbit, rel_vel, gamma_rv, u1, u3, gamma_u):
    big_bracket = - gamma_u * gamma_rv * rel_vel + gamma_u * u3 * (1 + (gamma_rv**2 * rel_vel**2) / (1 + gamma_rv))

    factor_1 = gamma_orbit * (gamma_u * gamma_rv - gamma_u * u3 * gamma_rv * rel_vel) - \
               gamma_orbit * orbit_vel * big_bracket
    factor_1 *= p0

    factor_2 = - gamma_u * u1
    factor_2 *= p1

    factor_3 = - gamma_orbit * orbit_vel * (gamma_u * gamma_rv - gamma_u * u3 * gamma_rv * rel_vel) + \
               (1 + (gamma_orbit**2 * orbit_vel**2) / (1 + gamma_orbit)) * big_bracket
    factor_3 *= p3

    return factor_1 + factor_2 + factor_3