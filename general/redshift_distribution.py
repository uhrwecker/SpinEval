import numpy as np
import matplotlib.pyplot as pl

import matplotlib as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_redshift_distribution(fp, ax, s, norm_color=(0, 0)):
    data = np.loadtxt(fp + '/redshift.csv', delimiter=',', skiprows=1)
    al = data[:, 0]
    be = data[:, 1]

    g = data[:, 2]

    g[g == 0] = np.nan

    n = int(np.sqrt(len(g)))
    g = g.reshape(n, n).T[::-1] - 1

    print(np.nanmin(g), np.nanmax(g))

    normx, normy = norm_color
    if not normx:
        normx = np.nanmin(g)
    if not normy:
        normy = np.nanmax(g)

    cmap = pl.cm.coolwarm.reversed()
    norm = mp.colors.Normalize(normx, normy)

    im = ax.imshow(g, extent=(np.amin(data[:, 0]), np.amax(data[:, 0]),
                   np.amin(data[:, 1]), np.amax(data[:, 1])), norm=norm, cmap=cmap)

    ax.scatter(0, 0, label=f's = {s}', s=0)
    ax.legend()
    ax.set_xlim(np.amin(data[:, 0]), np.amax(data[:, 0]))
    ax.set_ylim(np.amin(data[:, 1]), np.amax(data[:, 1]))

    return im


def main():
    phi = 0.0
    fps = [(f'Z:/Data/06022023/bigger_sample/bigger_sample/{phi}/', 0.00175),
           (f'Z:/Data/06022023/bigger_sample/bigger_sample/{phi}/', 0.00175),
           (f'Z:/Data/06022023/bigger_sample/bigger_sample/{phi}/', 0.00175),
           (f'Z:/Data/06022023/bigger_sample/bigger_sample/{phi}/', 0.00175),
           (f'Z:/Data/06022023/bigger_sample/bigger_sample/{phi}/', 0.00175),
           (f'Z:/Data/06022023/bigger_sample/bigger_sample/{phi}/', 0.00175),
           (f'Z:/Data/06022023/bigger_sample/bigger_sample/{phi}/', 0.00175),
           (f'Z:/Data/06022023/bigger_sample/bigger_sample/{phi}/', 0.00175),
           (f'Z:/Data/06022023/bigger_sample/bigger_sample/{phi}/', 0.00175)
            ]

    fig, axes = pl.subplots(3, 3, figsize=(11, 10), sharex=True, sharey=True)

    for fp, ax in zip(fps, axes.flatten()):
        fp0, s = fp

        im = plot_redshift_distribution(fp0, ax, s)

        if fp == fps[2] or fp == fps[5] == fps[8]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.05)

            fig.colorbar(im, cax=cax, orientation='vertical')

        if fp == fps[0] or fp == fps[3] or fp == fps[6]:
            ax.set_ylabel(r'$\beta$')

        if fp == fps[6] or fp == fps[7] or fp == fps[8]:
            ax.set_xlabel(r'$\alpha$')

    fig.set_tight_layout(True)
    pl.show()


if __name__ == '__main__':
    main()