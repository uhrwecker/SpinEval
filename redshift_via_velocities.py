import numpy as np
import matplotlib.pyplot as pl

from util import g

import matplotlib as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot(fp, label, ax, fig, flag=False):
    data = np.loadtxt(fp + '/redshift.csv', delimiter=',', skiprows=1)
    al = data[:, 0]
    be = data[:, 1]
    g = data[:, 2]
    g[g == 0] = np.nan

    n = int(np.sqrt(len(g)))
    g = g.reshape(n, n).T[::-1] - 1

    print(np.nanmin(g), np.nanmax(g))

    cmap = pl.cm.coolwarm.reversed()
    norm = mp.colors.Normalize(0.52-1, 1.2-1)

    im = ax.imshow(g, extent=(np.amin(data[:, 0]), np.amax(data[:, 0]),
                              np.amin(data[:, 1]), np.amax(data[:, 1])), norm=norm, cmap=cmap, interpolation='nearest')
    #im = ax.contourf(g, extent=(np.amin(data[:, 0]), np.amax(data[:, 0]),
    #                            np.amin(data[:, 1]), np.amax(data[:, 1])), norm=norm, cmap=cmap, vmin=0.52, vmax=1.2)

    if flag:
        print('huh')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)

        fig.colorbar(im, cax=cax, orientation='vertical')

    ax.scatter(0, 0, s=0, label=label)
    ax.legend()
    ax.set_xlim(-0.01, 0.01)#np.amin(data[:, 0]), np.amax(data[:, 0]))
    ax.set_ylim(np.amin(data[:, 1]), np.amax(data[:, 1]))
    ax.set_xlabel(r'$\alpha$')


def main():
    fig, ((ax1, ax2), (ax3, ax4)) = pl.subplots(2, 2, figsize=(7, 7), sharey=True, sharex=True)

    fps = [('/media/jan-menno/T7/Schwarzschild/influence_of_velocities/no_vel/0.0', 'redshift only', ax1),
           ('/media/jan-menno/T7/Schwarzschild/influence_of_velocities/orbit_vel/0.0', 'redshift + orbit velocity', ax2),
           ('/media/jan-menno/T7/Schwarzschild/influence_of_velocities/rel_vel/0.0', 'redshift + orbit + rel. velocity', ax3),
           ('/media/jan-menno/T7/Schwarzschild/influence_of_velocities/all_vel/0.0', 'all velocities', ax4)]


    for fp, label, ax in fps:
        flag = False
        if fp == '/media/jan-menno/T7/Schwarzschild/influence_of_velocities/all_vel/0.0' or fp == '/media/jan-menno/T7/Schwarzschild/influence_of_velocities/orbit_vel/0.0':
            flag = True
        plot(fp, label, ax, fig, flag)

    fig.set_tight_layout(True)

    ax1.set_ylabel(r'$\beta$')

    pl.show()


if __name__ == '__main__':
    main()
