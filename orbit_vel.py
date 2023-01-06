import numpy as np
import matplotlib.pyplot as pl

from util import orbit_velocity


def main():
    r = 8.
    s = np.linspace(-5e-3, 5e-3, num=100000)

    fig = pl.figure(1, figsize=(7, 4))
    ax = pl.gca()

    ax.plot(s, orbit_velocity(s, r), label='orbit velocity at r = 8M')
    ax.fill_between(np.linspace(-2e-3, 2e-3), 0, 1, color='DEEPSKYBLUE', alpha=0.2,
                    label='allowed region of spin')
    ax.set_xlim(-5e-3, 5e-3)
    ax.set_ylim(0.408, 0.4085)
    ax.set_xlabel('s')
    ax.set_ylabel(r'$v^3$')

    ax.legend()
    ax.grid()

    pl.show()


if __name__ == '__main__':
    main()