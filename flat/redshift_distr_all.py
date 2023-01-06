import numpy as np
import matplotlib.pyplot as pl

import matplotlib as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util import retrieve_north_pole, retrieve_eq


def plot_redshift_distribution(fp, ax, s, fig, flag=False):

    data = np.loadtxt(fp + '/redshift.csv', delimiter=',', skiprows=1)
    al = data[:, 0]
    be = data[:, 1]
    g = data[:, 2]
    g[g == 0] = np.nan

    a_gmin = al[g == np.nanmin(g)]
    b_gmin = be[g == np.nanmin(g)]
    #ax.scatter(a_gmin, b_gmin, s=10, color='green')

    a_gmax = al[g == np.nanmax(g)]
    b_gmax = be[g == np.nanmax(g)]
    #ax.scatter(a_gmax, b_gmax, s=10, color='green')

    n = int(np.sqrt(len(g)))
    g = g.reshape(n, n).T[::-1]
    if np.nanmin(g) < 0:
        g *= -1
    #g[np.isnan(g)] = 0

    print(np.nanmin(g), np.nanmax(g))

    cmap = pl.cm.YlOrRd
    norm = mp.colors.Normalize(0., 2.7)
    #norm = mp.colors.Normalize(np.nanmin(g), np.nanmax(g))

    margin = 0.6
    im = ax.imshow(g, extent=(np.amin(data[:, 0])-margin, np.amax(data[:, 0])+margin,
                   np.amin(data[:, 1])-margin, np.amax(data[:, 1])+margin), norm=norm, cmap=cmap)

    if s == 0.001 or s == -0.0005 or s == -0.00175 or flag:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)

        #fig.colorbar(im, cax=cax, orientation='vertical')

    if s == 0.00175 or s == 0.0005 or s == -0.001:
        ax.set_ylabel(r'$\beta$')

    if s == -0.001 or s == -0.0015 or s == -0.00175:
        ax.set_xlabel(r'$\alpha$')

    ax.scatter(0, 0, label=f's = {s}', s=0)
    #ax.legend()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    #ax.set_xlim(np.amin(data[:, 0]), np.amax(data[:, 0]))
    #ax.set_ylim(np.amin(data[:, 1]), np.amax(data[:, 1]))

    return np.nanmin(g), np.nanmax(g)

def main():
    import os

    fps = '/media/jan-menno/T7/Flat/analy/'

    fig, ax = pl.subplots(1, 1, figsize=(10, 10))
    phis = [x[0] for x in os.walk(fps)]
    phis = [phi for phi in phis if not phi.endswith('data')]
    phis = [phi for phi in phis if not phi == fps]
    phis.sort()

    ax.set_xlabel(r'$ \alpha $')
    ax.set_ylabel(r'$\beta$')
    #ax.scatter(0, 0, s=0, label='s =  0.0005')
    #ax.legend()
    print(phis)
    gmax = []
    gmin = []
    gmean = []

    for n, phi in enumerate(phis):
        ff = phi
        if n == 0:
            gn, gx = plot_redshift_distribution(ff, ax, '', fig, flag=True)
            pl.savefig(f'/media/jan-menno/T7/Flat/images/{n}.png')

        elif n % 3 == 0 or n == len(phis):
            gn, gx = plot_redshift_distribution(ff, ax, '', fig, flag=False)
            pl.savefig(f'/media/jan-menno/T7/Flat/images/{n}.png')

        else:
            continue
        gmax.append(gx)
        gmin.append(gn)
        gmean.append(np.abs(gx+ gn) / 2)
        fig.set_tight_layout(True)

        #pl.show()
        #pl.savefig(f'/media/jan-menno/T7/Flat/images/{n}.png')

        #ax.clear()
        #pl.show()

    ax.clear()
    ax.plot(np.linspace(0, np.pi*2, num=len(gmean)), gmean)
    #ax.set_ylim(0.5, 1)
    print(np.nanmin(gmin), np.nanmax(gmax))
    print(gmean)
    pl.show()

if __name__ == '__main__':
    main()