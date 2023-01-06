import numpy as np
import matplotlib.pyplot as pl


def main():
    rho = np.linspace(0., 0.01, num=10000)
    s = 2 / 5 * rho

    fig = pl.figure(111, figsize=(7, 4))

    ax = pl.gca()

    ax.plot(rho, s, label=r's$_{max} (\rho / M)$')
    ax.axvline(0.005, color='black', lw=2, ls='--', label=r's$_{max}(0.005M) = 0.002$')

    ax.set_xlim(0, 0.01)
    ax.set_ylim(0, 0.005)

    ax.set_xlabel(r'$\rho$ / M')
    ax.set_ylabel(r's$_{max}$')

    ax.grid()
    ax.legend()

    pl.show()


if __name__ == '__main__':
    main()