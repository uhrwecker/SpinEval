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


def pol_angle_at_obs(angle, alpha, beta):
    denom = np.sqrt(alpha**2 + beta**2)

    pa = np.arccos((- alpha * np.cos(angle) + beta * np.sin(angle)) / denom)

    sinpa = np.arcsin(- (beta * np.cos(angle) + alpha * np.sin(angle)) / denom)

    if sinpa < 0:
        #print('huh')
        pa = 2 * np.pi - pa

    return pa


def polarization_angle(d0, d2, p2, p3, v, gamma, u1, u3, gamma2, pem):
    f0 = np.sqrt(d2**2 / (d0**2 - d2**2))
    f1 = 0 # per definition
    f2 = np.sqrt(1 + f0**2)
    f3 = 0 # s.o.

    c_u1u3 = gamma2 ** 2 * u1 * u3 / (1 + gamma2)

    fb0 = gamma2 * f0 - gamma2 * u1 * f1 - gamma2 * u3 * f3
    fb3 = -gamma2 * u3 * f0 + c_u1u3 * f1 + (1 + gamma2**2 * u3**2 / (1 + gamma2)) * f3

    f0a = gamma * fb0 - gamma * v * fb3
    f3a = -gamma * v * fb0 + gamma * fb3

    numerator = f2 * p2 + f3a * p3

    norm = np.sqrt(p2 ** 2 + p3 ** 2) * np.sqrt(f2**2 + f3a**2)

    angle = np.arccos(numerator / norm)
    if pem < 0:
        pem += 2 * np.pi

    sin_numerator = f2 * p3 - f3a * p2

    if sin_numerator / norm < 0:#np.arcsin(sin_numerator / norm) < 0:
    #if pem > np.pi:
        angle = 2 * np.pi - angle

    return angle


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
            # step 5.1: momenta
            p0 = config['MOMENTA']['p_0']
            p1 = config['MOMENTA']['p_1']
            p2 = -config['MOMENTA']['p_2']
            p3 = config['MOMENTA']['p_3']

            # step 5.2: velocities
            d0 = 1.
            d1 = config['INITIAL_DATA']['dr']
            d2 = -config['INITIAL_DATA']['dtheta']
            d3 = config['INITIAL_DATA']['dphi']

            d2 *= config['INITIAL_DATA']['r0']
            d3 *= config['INITIAL_DATA']['r0'] * np.sin(config['INITIAL_DATA']['theta0'])

            rho = config['EMITTER']['rho']
            T = config['EMITTER']['Theta']
            P = config['EMITTER']['Phi']

            alpha = config['OBSERVER']['alpha']
            beta = config['OBSERVER']['beta']

            # step 6a: setup velocities
            surf = SurfaceVelocityRigidSphere(s, [rho, T, P])

            # step 6b: eval velocities
            v3 = 0.5
            gv3 = 1 / np.sqrt(1 - v3 ** 2)  # (v3, ), gv3 = orbit.get_velocity()
            # (rv, ), grv = rel_vel.get_velocity()
            rv = 0.0
            grv = 1.0
            (u1, u3), gu = surf.get_velocity()

            angle = polarization_angle(d0, d2, p2, p3, v3, gv3, u1, u3, gu, config['INITIAL_DATA']['phi0'])
            angle2 = pol_angle_at_obs(angle, alpha, beta)

            if angle > np.pi:
                angle -= 2 * np.pi

            if angle2 > np.pi / 2:
                angle2 += 2 * np.pi

            result_1.append(np.nanmax(angle / np.pi * 180))
            result_2.append(np.nanmin(angle2 / np.pi * 180))
            result_3.append(np.nanmin(np.sqrt(p2 ** 2 + p3 ** 2)))

            break

        # step 9a: save row
        row[-1] = angle
        redshift_data[n] = row

        # step 9b: remove file from list
        if not filenames == []:
            filenames.remove(file)

    # step 10: save pol angle on surface
    np.savetxt(fp_to_new_redshift + 'polarization2.csv', redshift_data, delimiter=',', header='alpha,beta,redshift')

    return np.nanmin(result_1), np.mean(result_1), np.nanmean(result_3)


def main(fp_to_data, fp_to_save):
    s = -0.0005#9
    rem = 8.0

    phis = [x[0] for x in os.walk(fp_to_data)]
    phis = [phi for phi in phis if not (phi.endswith('data') or phi.endswith('extra'))]
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
    pl.plot(ps, res2)
    #pl.plot(ps, res3)
    #pl.axhline(np.nanmax(imax) - np.abs(np.nanmax(imax) - np.nanmin(imin)) / 2, lw='1', ls='--')
    pl.xlim(0, 360)
    pl.xlabel(r'$\varphi_{em}$' + ' in degrees')
    pl.ylabel(r'$\psi$' + ' in degrees')
    pl.grid()

    df = pd.DataFrame({'phi': ps, 'psi': res1})
    #df.to_csv('/media/jan-menno/T7/Flat/data_v09.csv')
    pl.show()


if __name__ == '__main__':
    fp_to_data = '/media/jan-menno/T7/Flat/v05/s0/'
    fp_to_save = '/media/jan-menno/T7/Flat/pol_at_obs/s-005/'
    main(fp_to_data, fp_to_save)


