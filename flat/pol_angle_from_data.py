import numpy as np
import matplotlib.pyplot as pl


def main():
    data_0 = np.loadtxt('/media/jan-menno/T7/Flat/data_v0.csv', skiprows=1, delimiter=',')
    data_01 = np.loadtxt('/media/jan-menno/T7/Flat/data_v01.csv', skiprows=1, delimiter=',')
    data_05 = np.loadtxt('/media/jan-menno/T7/Flat/data_v05.csv', skiprows=1, delimiter=',')
    data_09 = np.loadtxt('/media/jan-menno/T7/Flat/data_v09.csv', skiprows=1, delimiter=',')

    pl.figure(figsize=(13, 6))

    #data_0[:, 1][-1] = 2 * np.pi
    #data_01[:, 1][-1] = 2 * np.pi
    #data_05[:, 1][-1] = 2 * np.pi
    #data_09[:, 1][-1] = 2 * np.pi

    pl.plot(data_0[:, 1], data_0[:, 2], label='v = 0.')
    pl.plot(data_01[:, 1], data_01[:, 2], label='v = 0.1')
    pl.plot(data_01[:, 1], data_05[:, 2], label='v = 0.5')
    pl.plot(data_01[:, 1], data_09[:, 2], label='v = 0.9')
    pl.xlim(0, 360)
    pl.xlabel(r'$\varphi_{obs}$ in degrees')
    #pl.ylim(0, 360)
    pl.ylabel(r'$\psi$ in degrees')
    pl.grid()
    pl.legend()

    pl.show()


if __name__ == '__main__':
    main()