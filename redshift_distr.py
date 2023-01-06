import numpy as np
import matplotlib.pyplot as pl

import matplotlib as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util import retrieve_north_pole, retrieve_eq


def plot_redshift_distribution(fp, ax, s, fig):

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
    #g[np.isnan(g)] = 0

    print(np.nanmin(g), np.nanmax(g))

    cmap = pl.cm.coolwarm.reversed()
    norm = mp.colors.Normalize(0.24163192789876406, 1.5244882152374375)
    norm = mp.colors.Normalize(-0.75, 0.75)

    im = ax.imshow(g, extent=(np.amin(data[:, 0]), np.amax(data[:, 0]),
                   np.amin(data[:, 1]), np.amax(data[:, 1])), norm=norm, cmap=cmap)
    #im = ax.pcolormesh(np.linspace(np.amin(data[:, 0]), np.amax(data[:, 0]), num=n),
    #                   np.linspace(np.amin(data[:, 1]), np.amax(data[:, 1]), num=n),
    #                   g, cmap=cmap, norm=norm, shading='gouraud', aa=True)

    if s == 0.001 or s == -0.0005 or s == -0.00175:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)

        fig.colorbar(im, cax=cax, orientation='vertical')

    if s == 0.00175 or s == 0.0005 or s == -0.001:
        ax.set_ylabel(r'$\beta$')

    if s == -0.001 or s == -0.0015 or s == -0.00175:
        ax.set_xlabel(r'$\alpha$')

    ax.scatter(0, 0, label=f's = {s}', s=0)
    ax.legend()
    ax.set_xlim(np.amin(data[:, 0]), np.amax(data[:, 0]))
    ax.set_ylim(np.amin(data[:, 1]), np.amax(data[:, 1]))


def main():
    phi = 0.0
    fps = [(f'/media/jan-menno/T7/Schwarzschild/redshift_dist_0_sphere/s0175/{phi}/', 0.00175),
           #(f'/home/jan-menno/Data/10_12_21/s015/{phi}/', 0.0015),
           (f'/media/jan-menno/T7/Schwarzschild/redshift_dist_0_sphere/s01/{phi}/', 0.001),
           #(f'/home/jan-menno/Data/10_12_21/s005/{phi}/', 0.0005),
           (f'/media/jan-menno/T7/Schwarzschild/redshift_dist_0_sphere/s0/{phi}/', 0.),
           #(f'/home/jan-menno/Data/10_12_21/s-005/{phi}/', -0.0005),
           (f'/media/jan-menno/T7/Schwarzschild/redshift_dist_0_sphere/s-01/{phi}/', -0.001),
           #(f'/home/jan-menno/Data/10_12_21/s-015/{phi}/', -0.0015),
           (f'/media/jan-menno/T7/Schwarzschild/redshift_dist_0_sphere/s-0175/{phi}/', -0.00175)]

    #fig, axes = pl.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
    fig = pl.figure(figsize=(7, 10))
    ax1 = pl.subplot2grid((6, 4), [0, 0], 2, 2)
    ax2 = pl.subplot2grid((6, 4), [0, 2], 2, 2)
    ax3 = pl.subplot2grid((6, 4), [2, 1], 2, 2)
    ax4 = pl.subplot2grid((6, 4), [4, 0], 2, 2)
    ax5 = pl.subplot2grid((6, 4), [4, 2], 2, 2)

    axes = np.array([ax1, ax2, ax3, ax4, ax5])

    smallest, ab = retrieve_north_pole(f'/media/jan-menno/T7/Schwarzschild/redshift_dist_0_sphere/s015/{phi}/data/')
    #eq = retrieve_eq(f'/home/jan-menno/Data/10_12_21/s-015/{phi}/data/')
    for fp, ax in zip(fps, axes.flatten()):
        if fp:
            plot_redshift_distribution(fp[0], ax, fp[1], fig)
        ax.scatter(ab[0], ab[1], c='black', s=10, marker='x')
        #for data in eq:
        #    ax.scatter(data[0],data[1], c='black', s=1)
    fig.set_tight_layout(True)
    #pl.savefig('/home/jan-menno/Data/image.png')
    pl.show()


if __name__ == '__main__':
    main()