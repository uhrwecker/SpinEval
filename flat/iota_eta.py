import numpy as np


def get_emission_angles(config, v3, gamma, u1, u3, gamma2, p0, p1, p2, p3):
    rem = config['INITIAL_DATA']['r0']
    pem = config['INITIAL_DATA']['phi0']
    if pem < 0:
        pem += np.pi * 2

    dr = config['INITIAL_DATA']['dr']
    factor = 1
    p0 = 1. * factor
    p1 = config['MOMENTA']['p_1'] * factor
    p2 = config['MOMENTA']['p_2'] * factor
    #p2 /= config['INITIAL_DATA']['r0']
    p3 = config['MOMENTA']['p_3'] * factor

    #p3 /= (config['INITIAL_DATA']['r0'] * np.sin(config['INITIAL_DATA']['theta0']))

    #const_v = (1 + gamma ** 2 * v3 ** 2 / (1 + gamma))
    #chi = gamma2 * (gamma * p0 + gamma * v3 * p3) + gamma2 * u1 * p1 + gamma2 * u3 * (gamma * v3 * p0 + const_v * p3)

    eta = np.arccos(p2)
    #iota = calc_iota(rem, dr, p3, eta, v3, gamma, u1, u3, gamma2)
    iota = calc_iota2(p0, p1, p2, p3, eta, gamma, v3, gamma2, u1, u3, pem)

    return iota, eta


def calc_iota2(p0, p1, p2, p3, eta, gamma, v3, gamma2, u1, u3, pem):
    const_v = gamma
    const_u1u3 = (gamma2 ** 2 * u3 * u1 / (1 + gamma2))

    cos_nom = gamma2 * u1 * (gamma * p0 + gamma * v3 * p3) \
            + (1 + gamma2 ** 2 * u1 ** 2 / (1 + gamma)) * p1 \
            + const_u1u3 * (gamma * v3 * p1 + const_v * p3)

    sin_nom = gamma2 * u3 * (gamma * p0 + gamma * v3 * p3) \
            + const_u1u3 * p1 \
            + (1 + gamma2 ** 2 * u3 ** 2 / (1 + gamma)) * (gamma * v3 * p1 + const_v * p3)

    iota = np.arccos(cos_nom / (p2 * np.tan(eta)))

    if pem < 0:
        pem += np.pi * 2

    sin = sin_nom / np.sin(eta)#cos_nom

    #if sin < -1:
    #    print(sin)#sin += 2

    if pem > np.pi:
        #print(sin)
    #    if pem > np.pi:
    #        print(sin)
        iota = 2 * np.pi - iota

    return iota % (2 * np.pi)


def calc_iota(rem, dr, dphi, eta, v3, gamma, u1, u3, gamma2):
    p0 = 1
    c = (1 + gamma**2 * v3**2 / (1 + gamma))
    k1 = gamma * v3 * gamma2 * u1 + c * (gamma2 ** 2 * u3 * u1 / (1 + gamma2))
    k2 = gamma * v3 * gamma2 * u3 + c * (1 + gamma2 ** 2 * u3 ** 2 / (1 + gamma2))
    k3 = 1 + gamma2 ** 2 * u1 ** 2 / (1 + gamma2)
    k4 = gamma2 ** 2 * u1 * u3 / (1 + gamma2)

    denom = np.sin(eta) * (k1 * k4 - k2 * k3)

    phi_term = dphi - gamma * v3 * gamma2 + gamma2 * u3 * c
    r_term = dr + gamma2 * u1

    cos = (k4 * phi_term - k2 * r_term) / denom

    iota = np.arccos(cos)
    #if iota < 0:
    #    iota += np.pi * 2

    sin = -(k3 * phi_term - k1 * r_term) / denom
    print(sin)
    iota_2 = np.arcsin(sin)

    if sin < 0:
        iota = 2 * np.pi - iota

    return iota % (np.pi * 2)