import numpy as np
import matplotlib.pyplot as pl

import matplotlib as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util import retrieve_north_pole, retrieve_eq


def draw_arrows(ax, g, alpha, beta, margin):
    interval = 129
    n_sample = np.arange(0, len(g))[::interval]
    g_sample = g[::interval]

    alpha += np.abs(np.amax(alpha) - np.amin(alpha)) / int(np.sqrt(len(g)))
    # beta += np.abs(np.amax(beta) - np.amin(beta)) / int(np.sqrt(len(g)))

    factor = 0.000066  # 18

    for n, sample in zip(n_sample, g_sample):
        marg = 2 * margin * n / len(n_sample) - margin
        if np.isnan(sample):
            continue

        sample -= np.pi / 2
        if sample < 0:
            sample += 2 * np.pi

        a = alpha[n]  # + marg
        b = beta[n]  # + marg
        ax.arrow(a, b, factor * np.sin(sample), -factor * np.cos(sample), head_length=0.,
                 head_width=0.)
        # ax.annotate('', xy=(a, b), xytext=(a+factor*np.cos(sample), b+factor*np.sin(sample)), arrowprops=dict(arrowstyle='->'))


def plot_redshift_distribution(fp, ax, s, fig, flag=False):
    data = np.loadtxt(fp + '/polarization.csv', delimiter=',', skiprows=1)
    al = data[:, 0]
    be = data[:, 1]
    g = data[:, 2]
    g[g == 0] = np.nan

    g += 1.5 * np.pi

    g = g % (2 * np.pi)

    draw_arrows(ax, g, data[:, 0], data[:, 1], margin=0.6)

    # g *= 180 / np.pi
    a_gmin = al[g == np.nanmin(g)]
    b_gmin = be[g == np.nanmin(g)]
    # ax.scatter(a_gmin, b_gmin, s=10, color='green')

    a_gmax = al[g == np.nanmax(g)]
    b_gmax = be[g == np.nanmax(g)]
    # ax.scatter(a_gmax, b_gmax, s=10, color='green')

    n = int(np.sqrt(len(g)))
    g = g.reshape(n, n).T[::-1]
    # if np.nanmin(g) < 0:
    #    g *= -1
    # g[np.isnan(g)] = 0

    print(np.nanmin(g), np.nanmax(g))

    cmap = pl.cm.hsv
    norm = mp.colors.Normalize(0, 2 * np.pi)
    # norm = mp.colors.Normalize(np.nanmin(g), np.nanmax(g))

    margin = 0.
    im = ax.imshow(g, extent=(np.amin(data[:, 0]) - margin, np.amax(data[:, 0]) + margin,
                              np.amin(data[:, 1]) - margin, np.amax(data[:, 1]) + margin), norm=norm, cmap=cmap)

    # if s == 0.001 or s == -0.0005 or s == -0.00175 or flag:
    #    divider = make_axes_locatable(ax)
    #    cax = divider.append_axes('right', size='2%', pad=0.05)

    # fig.colorbar(im, cax=cax, orientation='vertical')

    if s == 0.00175 or s == 0.0005 or s == -0.001:
        ax.set_ylabel(r'$\beta$')

    if s == -0.001 or s == -0.0015 or s == -0.00175:
        ax.set_xlabel(r'$\alpha$')

    ax.scatter(0, 0, label=f's ={s}', s=0)
    ax.legend()
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)
    ax.set_xlim(np.amin(data[:, 0]), np.amax(data[:, 0]))
    ax.set_ylim(np.amin(data[:, 1]), np.amax(data[:, 1]))

    return np.nanmin(g), np.nanmax(g), im


def main(m):
    import os

    base = 'higher_resolution/pol_angle_0_sphere_2'

    fps = [   f'/media/jan-menno/T7/Schwarzschild/{base}/s-0175/',
         f'/media/jan-menno/T7/Schwarzschild/{base}/s-015/',
         f'/media/jan-menno/T7/Schwarzschild/{base}/s-01/',
         f'/media/jan-menno/T7/Schwarzschild/{base}/s-005/',
        f'/media/jan-menno/T7/Schwarzschild/{base}/s0/',
         f'/media/jan-menno/T7/Schwarzschild/{base}/s005/',
         f'/media/jan-menno/T7/Schwarzschild/{base}/s01/',
         f'/media/jan-menno/T7/Schwarzschild/{base}/s015/',
         f'/media/jan-menno/T7/Schwarzschild/{base}/s0175/'
    ]

    ss = ['-0.00175', '-0.00150', '-0.00100', '-0.00050', ' 0.00000', ' 0.00050', ' 0.00100', ' 0.00150', ' 0.00175']
    # ss = ['-0.00175', '-0.00100', ' 0.00000', ' 0.00100', ' 0.00175']
    #ss = ['-0.0', '0.15']

    #fig = pl.figure(figsize=(7, 10))
    #ax1 = pl.subplot2grid((6, 4), [0, 0], 2, 2)
    #ax2 = pl.subplot2grid((6, 4), [0, 2], 2, 2)
    #ax3 = pl.subplot2grid((6, 4), [2, 1], 2, 2)
    #ax4 = pl.subplot2grid((6, 4), [4, 0], 2, 2)
    #ax5 = pl.subplot2grid((6, 4), [4, 2], 2, 2)

    #axes = np.array([ax1, ax2, ax3, ax4, ax5])
    #axes = np.array([ax3])
    fig, ax = pl.subplots(3, 3, figsize=(11, 13), sharex=True, sharey=True)
    #fig, ax = pl.subplots(1, 1, figsize=(10, 10))
    #axes = np.array([ax])
    axes = ax.flatten()

    for ax, fp, s in zip(axes, fps, ss):
        print(fp)
        phis = [x[0] for x in os.walk(fp)]
        phis = [phi for phi in phis if not phi.endswith('data')]
        phis = [phi for phi in phis if not phi == fp]
        phis.sort()

        if fp in [f'/media/jan-menno/T7/Schwarzschild/{base}/s01/',
                  f'/media/jan-menno/T7/Schwarzschild/{base}/s015/',
                  f'/media/jan-menno/T7/Schwarzschild/{base}/s0175/']:
            ax.set_xlabel(r'$ \alpha $')
        if fp in [f'/media/jan-menno/T7/Schwarzschild/{base}/s-0175/',
                  f'/media/jan-menno/T7/Schwarzschild/{base}/s-005/',
                  f'/media/jan-menno/T7/Schwarzschild/{base}/s01/']:
            ax.set_ylabel(r'$\beta$')
        # ax.scatter(0, 0, s=0, label='s =  0.0005')
        # ax.legend()
        # print(phis)
        gmax = []
        gmin = []
        gmean = []

        for n, phi in enumerate(phis):
            ff = phi
            if n == m:
                print(ff)
                gn, gx, im = plot_redshift_distribution(ff, ax, s, fig, flag=True)

                if fp in [f'/media/jan-menno/T7/Schwarzschild/{base}/s-01/',
                          f'/media/jan-menno/T7/Schwarzschild/{base}/s005/',
                          f'/media/jan-menno/T7/Schwarzschild/{base}/s0175/'] or len(axes) == 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='2%', pad=0.05)

                    fig.colorbar(im, cax=cax, orientation='vertical')

            # pl.savefig(f'/media/jan-menno/T7/Flat/images/i{n}.png')

            # elif n % 3 == 0 or n == len(phis):
            #    gn, gx, im = plot_redshift_distribution(ff, ax, '', fig, flag=False)
            #    pl.savefig(f'/media/jan-menno/T7/Flat/images/i{n}.png')

            else:
                continue
            gmax.append(gx)
            gmin.append(gn)
            gmean.append(np.abs(gx + gn) / 2)
            fig.set_tight_layout(True)

            # pl.show()
            # pl.savefig(f'/media/jan-menno/T7/Flat/images/{n}.png')

            # ax.clear()
            # pl.show()

            # ax.clear()
            # ax.plot(np.linspace(0, np.pi*2, num=len(gmean)), gmean)
            # ax.set_ylim(0.5, 1)
            print(np.nanmin(gmin), np.nanmax(gmax))
    print(gmean)
    pl.show()
    # pl.savefig(f'/media/jan-menno/T7/Flat/results/animation_pol/{m:03d}.png')


if __name__ == '__main__':
    for m in [0]:
        main(m)
