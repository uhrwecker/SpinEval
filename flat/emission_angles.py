import numpy as np
import os
import matplotlib.pyplot as pl
import json
from iota_eta import get_emission_angles
import pandas as pd

class VelocityABC:
    """Abstract base class for any velocity yet to come. Any velocity has to have these entry points."""
    def __init__(self, s, position):
        self.s = s
        self.position = position #might be any position (orbit distance, rel distance etc)

        self.vel = self._calculate_velocity()

    def _calculate_velocity(self):
        """This method is used to calculate the specific velocities
        and should ALWAYS ONLY use the class variables.
        Return:
            (Velocity, ..) AND Gamma = 1 / 1 - sqrt(all_v**2) """

        raise NotImplementedError('You have to build your velocity class correctly bro.')

    def get_velocity(self):
        return self.vel

    def change_position(self, position, recalc=True):
        self.position = position

        if recalc:
            self.vel = self._calculate_velocity()


class SurfaceVelocityRigidSphere(VelocityABC):
    """Surface velocities u1 and u3 of a perfect rigid sphere."""
    def __init__(self, s, position):
        super().__init__(s, position)

        # note that position should be an iterable with
        # position = (rho, theta, phi)

    def _calculate_velocity(self):
        rho, theta, phi = self.position

        u1 = 5 * self.s / (2 * rho) * np.sin(phi) * np.sin(theta)
        u3 = 5 * self.s / (2 * rho) * np.cos(phi) * np.sin(theta)

        if 1 - u1 ** 2 - u3 ** 2 < 0:
            print('Velocities too high; returning nan.')
            return np.nan, np.nan, np.nan

        return (-u1, -u3), 1 / np.sqrt(1 - u1 ** 2 - u3 ** 2)


def convert_p(p0, p1, p3, v3, gamma, u1, u3, gamma2):
    p00 = gamma * p0 - gamma * v3 * p3
    p33 = -gamma * v3 * p0 + (1 + gamma ** 2 * v3 ** 2) * p3

    p333 = -gamma2 * u3 * p00 + (gamma2 ** 2 * u1 * u3) / (1 + gamma2) * p1 \
           + (1 + gamma2 ** 2 * u3**2 / (1 + gamma2)) * p33

    p000 = gamma2 * p00 - gamma2 * u1 * p1 - gamma2 * u3 * p33

    return p000, p333


def polarize(p00, p22, p33, p2, p3, v3, gamma, u1, u3, gamma2):
    K = -(gamma * gamma2 * v3 * u3 + gamma * (1 + gamma2**2 * u3**2 / (1 + gamma2)))
    root = p00**2 * (p00**2 - p22 ** 2) * (p2**2 + p3**2)

    a = - p3 * np.sqrt(p00 ** 2 - p22 ** 2) / (p00 * np.sqrt(p2**2 + p3 ** 2))
    b = p2 * K * p22 * p33 / (p00 * np.sqrt(p2**2 + p3**2) * np.sqrt(p00**2 - p22**2))

    if np.isnan(a) or np.isnan(b):
        print(a, b)

    if a + b > 1 or a + b < -1:
        print(a + b)

    #if a + b > 1:
    #    a = 1 - b

    #if a + b < -1:
    #    a = - b - 1

    return a + b

    #eturn (-p3 * np.sin(eta) + K * p2 * np.cos(eta) * p33) / np.sqrt(root)
    return (-p3 * (p00 ** 2 - p22 ** 2) + K * p2 * p22 * p33) / np.sqrt(root)


def polarize2(iota, eta, p2, p3, v3, gamma, u1, u3, gamma2):
    K = - (gamma * gamma2 * v3 * u3 + (1 + gamma**2 * v3**2 / (1 + gamma)) * (1 + gamma2**2 * u3**2 / (1 + gamma2)))

    root = (np.sin(eta)) * (p2**2 + p3**2)

    p00 = 1.
    p22 = np.cos(eta)
    p33 = np.sin(iota) * np.sin(eta)

    a = - p3 * np.sqrt(np.sin(eta)**2) / (p00 * np.sqrt(p2 ** 2 + p3 ** 2))
    b = p2 * K * p22 * p33 / (p00 * np.sqrt(p2 ** 2 + p3 ** 2) * np.sqrt(np.sin(eta)**2))

    return a + b

    return (-p3 * np.sin(eta) + K * p2 * np.cos(eta) * np.sin(iota) * np.sin(eta)) / np.sqrt(root)


def alternative_velocity(config):
    robs = 35.
    tobs = 1.
    pobs = 0.

    rem = config['INITIAL_DATA']['r0']
    tem = config['INITIAL_DATA']['theta0']
    pem = config['INITIAL_DATA']['phi0']

    xobs = robs * np.sin(tobs)
    yobs = 0.
    zobs = robs * np.cos(tobs)

    xem = rem * np.cos(pem) * np.sin(tem)
    yem = rem * np.sin(pem) * np.sin(tem)
    zem = rem * np.cos(tem)

    S = xobs * xem + yobs * yem + zobs * zem

    dt = robs
    dr = (S - rem ** 2) / rem
    dtheta = (robs * np.cos(tobs) - S * np.cos(tem)) / (rem * np.sin(tem))
    dphi = robs / rem * np.sin(-pem)

    return dt , dr  , dtheta , dphi


def evaluate(s, rem, fp_to_redshift, fp_to_new_redshift):
    # step 0:
    fp_to_json = fp_to_redshift + 'data/'

    # step 1: get a list of every json file in specified directory
    filenames = next(os.walk(fp_to_json), (None, None, []))[2]

    # step 2: load .csv of redshift
    redshift_data = np.loadtxt(fp_to_redshift + 'redshift.csv', delimiter=',', skiprows=1)

    result_1 = []
    result_2 = []
    result_3 = []

    # step 3: iterate over redshift data:
    for n, row in enumerate(redshift_data):
        # step 3a: exclude redshift data for when there is no collision:
        if row[-1] == 0.:
            continue

        # step 4: load all json data one by one:
        for file in filenames:
            with open(fp_to_json + file, 'r') as f:
                config = json.load(f)

            # step 4a: if the row does not equal the alpha/beta of json, delete config
            if config['OBSERVER']['alpha'] != row[0] or config['OBSERVER']['beta'] != row[1]:
                del config
                continue

            # step 5: read the relevant data
            p0 = config['MOMENTA']['p_0']
            p1 = config['MOMENTA']['p_1']
            p2 = config['MOMENTA']['p_2']
            p3 = config['MOMENTA']['p_3']

            #p0, p1, p2, p3 = alternative_velocity(config)
            p1 = config['INITIAL_DATA']['dr']
            p2 = config['INITIAL_DATA']['dtheta']
            p3 = config['INITIAL_DATA']['dphi']

            #p2 /= (config['INITIAL_DATA']['r0']**2)

            p2 *= config['INITIAL_DATA']['r0']
            p3 *= config['INITIAL_DATA']['r0'] * np.sin(config['INITIAL_DATA']['theta0'])

            #p0 = np.sqrt(p1 ** 2 + p2 ** 2 + p3 ** 2)
            #print(p1 ** 2 + p2 ** 1 + p3 ** 2, p0 ** 2)
            #p0 = np.sqrt(p1**2 + p2**2 + p3**2)



            rho = config['EMITTER']['rho']
            T = config['EMITTER']['Theta']
            P = config['EMITTER']['Phi']

            alpha = config['OBSERVER']['alpha']
            beta = config['OBSERVER']['beta']

            # step 6a: setup velocities
            surf = SurfaceVelocityRigidSphere(s, [rho, T, P])

            # step 6b: eval velocities
            v3 = 0.9
            gv3 = 1 / np.sqrt(1 - v3 ** 2)  # (v3, ), gv3 = orbit.get_velocity()
            # (rv, ), grv = rel_vel.get_velocity()
            rv = 0.0
            grv = 1.0
            (u1, u3), gu = surf.get_velocity()

            p5 = np.sqrt(1 + (p2 / p0)**2) * config['MOMENTA']['p_2'] / (np.sqrt(1 + (p2 / p0)**2) * np.sqrt(config['MOMENTA']['p_1']**2 +
                                                                                                             config['MOMENTA']['p_2'] ** 2 +
                                                                                                             config['MOMENTA']['p_3'] ** 2))

            iota, eta = get_emission_angles(config, v3, gv3, u1, u3, gu, p0, p1, p2, p3)
            if iota < 0:
                iota += np.pi * 2

            p00, p33 = convert_p(p0, p1, p3, v3, gv3, u1, u3, gu)

            # step 7: calculate redshift
            p = np.arccos(polarize(p00, p2, p33, p2, p3, v3, gv3, u1, u3, gu)) / np.pi * 180 - 90
            p2 = np.arccos(polarize2(iota, eta, p2, p3, v3, gv3, u1, u3, gu)) / np.pi * 180 - 90
            #if (config['INITIAL_DATA']['phi0'] + 2 * np.pi) % (np.pi *2) > 1.27:
            #    print(p0, p1, p2, p3)
            #    print(polarize(p00, p2, p33, p2, p3, v3, gv3, u1, u3, gu))
            #    print(np.arccos(polarize(p00, p2, p33, p2, p3, v3, gv3, u1, u3, gu)))
            #    raise KeyError

            #p_o = np.arccos(-(alpha * np.sin(p) + beta * np.cos(p))/np.sqrt(alpha**2 + beta**2)) / np.pi * 180 #- 90
            #p = np.arccos(polarize(p00, p2, p33, p2, p3, v3, gv3, u1, u3, gu)) / np.pi * 180 - 90

            dr = config['INITIAL_DATA']['dr']

            result_1.append(np.nanmax(iota))
            result_2.append(np.nanmin(eta))
            result_3.append(np.nanmin(p5))

            break

        # step 9a: save row
        row[-1] = p
        redshift_data[n] = row

        # step 9b: remove file from list
        if not filenames == []:
            filenames.remove(file)

    return np.nanmin(result_1), np.mean(result_2), np.nanmean(result_3)


def main(fp_to_data, fp_to_save):
    s = -0.00#1
    rem = 8.0

    phis = [x[0] for x in os.walk(fp_to_data)]
    phis = [phi for phi in phis if not phi.endswith('data')]
    phis = [phi + '/' for phi in phis if not phi == fp_to_data]
    phis.sort()

    res1 = []
    res2 = []
    res3 = []
    ps = []

    for n, file in enumerate(phis):
        phi = file[len(fp_to_data):-1]

        print(f'Now at {phi} ... ({n+1} / {len(phis)})')

        r1, r2, r3 = evaluate(s, rem, file, fp_to_save + phi + '/',)
        print(r1, r2, r3)

        res1.append(r1)
        res2.append(r2)
        res3.append(r3)
        ps.append(np.float(phi) / np.pi * 180)

    ps.append(360)
    res1.append(res1[0])
    res2.append(res2[0])
    res3.append(res3[0])

    pl.figure(figsize=(13, 9))
    #pl.plot(ps, res1)
    #pl.plot(ps, res2)
    pl.plot(ps, res3)
    #pl.axhline(np.nanmax(imax) - np.abs(np.nanmax(imax) - np.nanmin(imin)) / 2, lw='1', ls='--')
    pl.xlim(0, 360)
    pl.xlabel(r'$\varphi_{em}$' + ' in degrees')
    pl.ylabel(r'$\psi$' + ' in degrees')
    pl.grid()

    df = pd.DataFrame({'phi': ps, 'iota': res1, 'eta': res2, 'psi': res3})
    df.to_csv('/media/jan-menno/T7/Flat/data_v09.csv')
    pl.show()


if __name__ == '__main__':
    fp_to_data = '/media/jan-menno/T7/Flat/v05/s0/'
    fp_to_save = '/media/jan-menno/T7/Flat/pol/'
    main(fp_to_data, fp_to_save)


