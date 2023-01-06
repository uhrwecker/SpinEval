import numpy as np
import matplotlib.pyplot as pl

import matplotlib as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util import retrieve_north_pole, retrieve_eq


def plot_redshift_distribution(fp, ax, s, fig, m):
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
    g = g.reshape(n, n).T[::-1] - 1

    row = g[int(n/2) + 1]
    row = row[~np.isnan(row)]
    mid = int(len(row) / 2)
    g_mid = row[mid]
    a_mid, b_mid = al[:101][g[int(n/2) + 1] == g_mid][0], be[:101][g[int(n/2) + 1] == g_mid][0]

    #g[np.isnan(g)] = 0

    # draw the circle:
    circ_phi = np.linspace(0, 2*np.pi, num=1000)
    ax.plot(4.8*np.cos(circ_phi), 4.8*np.sin(circ_phi), color='black')

    print(np.nanmin(g), np.nanmax(g))

    cmap = pl.cm.coolwarm.reversed()
    norm = mp.colors.Normalize(0.24163192789876406, 1.5244882152374375)
    norm = mp.colors.Normalize(-0.4, 0.4)

    a_cent, b_cent = np.mean([np.amin(data[:, 0]), np.amax(data[:, 0])]), np.mean([np.amin(data[:, 1]), np.amax(data[:, 1])])
    a_dist, b_dist = np.abs(a_cent - np.amin(data[:, 0])), np.abs(b_cent - np.amin(data[:, 1]))
    factor = 40
    dis = 0.8
    amin, amax = np.amin(data[:, 0]) - factor * a_dist, np.amax(data[:, 0]) + factor * a_dist
    bmin, bmax = np.amin(data[:, 1]) - factor * b_dist, np.amax(data[:, 1]) + factor * b_dist

    #im = ax.imshow(g, extent=(np.amin(data[:, 0])-factor, np.amax(data[:, 0])+factor,
    #               np.amin(data[:, 1]) - factor, np.amax(data[:, 1]) + factor), norm=norm, cmap=cmap)
    im = ax.imshow(g, extent=(amin, amax, bmin, bmax), norm=norm, cmap=cmap)

    if m == 0:
        print('hi')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)

        fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    return np.mean([np.amin(data[:, 0]), np.amax(data[:, 0])]), np.mean([np.amin(data[:, 1]), np.amax(data[:, 1])])


def main():
    import os
    fps = '/media/jan-menno/T7/Schwarzschild/sphere/s0/'

    fig, ax = pl.subplots(1, 1, figsize=(10, 10))
    phis = [x[0] for x in os.walk(fps)]
    phis = [phi for phi in phis if not phi.endswith('data')]
    phis = [phi for phi in phis if not phi == fps]
    phis.sort()

    path_x = []
    path_y = []
    #fig, axes = pl.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)

    #eq = retrieve_eq(f'/home/jan-menno/Data/10_12_21/s-015/{phi}/data/')
    for m in range(0, 4):
        for n, phi in enumerate(phis):
            n += m * len(phis)

            #smallest, ab = retrieve_north_pole(
            #    f'{phi}/data/')
            if len(path_x) > 126:
                del path_x[0]
            if len(path_y) > 126:
                del path_y[0]
            ax.plot(path_x, path_y, alpha=0.5)
            x, y = plot_redshift_distribution(phi, ax, 0.0015, fig, n)
            path_x.append(x)
            path_y.append(y)

            #ax.scatter(ab[0], ab[1], c='black', s=10, marker='x')
        #for data in eq:
        #    ax.scatter(data[0],data[1], c='black', s=1)
            fig.set_tight_layout(True)
            pl.savefig(f'/media/jan-menno/T7/Schwarzschild/animation/{n:04d}.png')
            ax.clear()




if __name__ == '__main__':
    main()