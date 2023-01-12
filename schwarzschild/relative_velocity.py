import numpy as np
import matplotlib.pyplot as pl

from util import relative_velocity


def main():
    fig, (ax1, ax2) = pl.subplots(2, 1, figsize=(7, 7))

    # upper image
    r = np.linspace(2., 10, num=1000000)

    ax1.plot(r, relative_velocity(1, r), label='s =  1')
    ax1.plot(r, relative_velocity(-1, r), label='s = -1')
    ax1.fill_between(np.linspace(2, 3), -1, 2, color='DEEPSKYBLUE', alpha=0.2)
    ax1.axvline(8, lw=2, ls='--', c='black', label='emitter position')
    ax1.set_xlabel('r / M')
    ax1.set_ylabel(r'$\mathcal{V}$')
    ax1.set_xlim(2, 10)
    ax1.set_ylim(-1e-5, 1)
    ax1.grid()
    ax1.legend()

    # lower image
    s = np.linspace(-2e-1, 2e-1, num=1000000)
    r = 8.

    ax2.plot(np.linspace(-2e-3, 2e-3, num=1000000), relative_velocity(s, r)**2, label=r'$\mathcal{V}$ at r = 8M')
    ax2.set_xlabel('s')
    ax2.set_ylabel(r'$\mathcal{V}$')
    ax2.set_xlim(-2e-3, 2e-3)
    #ax2.set_ylim(0, 1e-14)
    ax2.grid()
    ax2.legend()

    pl.show()


if __name__ == '__main__':
    main()