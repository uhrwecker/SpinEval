import numpy as np
import os
import json
from scipy.optimize import root_scalar

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


def polarize(iota, eta, p2, p3, v3, gamma, u1, u3, gamma2):
    K = - (gamma * gamma2 * v3 * u3 + (1 + gamma**2 * v3**2 / (1 + gamma)) * (1 + gamma2**2 * u3**2 / (1 + gamma2)))

    root = np.sin(eta)**2 * (p2**2 + p3**2)

    return np.cos(-(p3 * np.sin(eta)**2 + K * p2 * np.cos(eta) * np.sin(iota) * np.sin(eta)) / np.sqrt(root))
    #return (p3 * np.sin(eta)**2 + K * p2 * np.cos(eta) * np.sin(iota) * np.sin(eta)) / np.sqrt(root)

def eval_spin_stuff(s, rem, fp_to_redshift, fp_to_new_redshift, robs=35., tobs=1.):
    # step 0:
    fp_to_json = fp_to_redshift + 'data/'

    # step 1: get a list of every json file in specified directory
    filenames = next(os.walk(fp_to_json), (None, None, []))[2]

    # step 2: load .csv of redshift
    redshift_data = np.loadtxt(fp_to_redshift + 'redshift.csv', delimiter=',', skiprows=1)

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
            p2 = config['MOMENTA']['p_2']
            p3 = config['MOMENTA']['p_3']

            rho = config['EMITTER']['rho']
            T = config['EMITTER']['Theta']
            P = config['EMITTER']['Phi']

            alpha = config['OBSERVER']['alpha']
            beta = config['OBSERVER']['beta']

            # step 6a: setup velocities
            #orbit = orbit_vel.OrbitVelocityKerr(s, a, rem)
            #rel_vel = relative_vel.RelativeVelocityKerr(s, a, rem)
            surf = SurfaceVelocityRigidSphere(s, [rho, T, P])

            # step 6b: eval velocities
            v3 = 0.5
            gv3 = 1 / np.sqrt(1 - 0.25)  # (v3, ), gv3 = orbit.get_velocity()
            # (rv, ), grv = rel_vel.get_velocity()
            rv = 0.0
            grv = 1.0
            (u1, u3), gu = surf.get_velocity()

            iota, eta = get_emission_angles(config, v3, gv3, u1, u3, gu)

            # step 7: calculate redshift
            p = polarize(iota, eta, p2, p3, v3, gv3, u1, u3, gu)
            #p = np.arccos((- alpha * np.cos(p) + beta * np.sin(p))/np.sqrt(alpha**2 + beta**2))

            # step 8: calc redshift at observer
            # delta = robs ** 2 - 2 * robs + a ** 2
            # A = (robs ** 2 + a ** 2) ** 2 - delta * a * np.sin(tobs) ** 2
            # omega = 2 * a * robs / A
            # e_min_nu = np.sqrt(A / ((robs ** 2 + a ** 2 * np.cos(tobs) ** 2) * delta))
            # g = 1 / g

            break

        # step 9a: save row
        row[-1] = p
        redshift_data[n] = row

        # step 9b: remove file from list
        if not filenames == []:
            filenames.remove(file)

    # step 10: save new redshift
    np.savetxt(fp_to_new_redshift + 'redshift.csv', redshift_data, delimiter=',', header='alpha,beta,redshift')


def get_emission_angles(config, v3, gamma, u1, u3, gamma2):
    rem = config['INITIAL_DATA']['r0']

    dr = config['INITIAL_DATA']['dr']
    dtheta = config['INITIAL_DATA']['dtheta']

    eta = np.arccos(rem * dtheta)

    def _iota_comp():
        a = (1 + gamma2**2 * u1**2 / (1 + gamma2))
        b = gamma2**2 * u1 * u3 / (1 + gamma2)
        c = (dr - gamma2 * u1) * np.sin(eta)

        return 2*np.arctan((b - np.sqrt(a**2 + b**2 - c**2)) / (a + c))
    #def func(iota):
    #    f1 = (1 + gamma2**2 * u1**2 / (1 + gamma2))
    #    f2 = gamma2**2 * u1 * u3 / (1 + gamma2)

    #    return f1 * np.cos(iota) + f2 * np.sin(iota) - (dr - gamma2 * u1) * np.sin(eta)

    #iota = root_scalar(func, x0=-np.pi, x1=np.pi/2)
    #print(iota.root % (np.pi * 2))
    #raise ValueError()

    return _iota_comp(), eta


