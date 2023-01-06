import numpy as np

def dphi(tau, rem, tem, pem, robs, tobs, pobs):
    g = robs * np.sin(pobs) - rem * np.sin(pem)
    h = rem * np.sin(pem)
    k = robs * np.cos(pobs) - rem * np.cos(pem)
    l = rem * np.cos(pem)

    return (g*l - k*h) / ((g * tau + h) ** 2 + (k * tau + l)**2)


def red(rem, tem, pem, robs, tobs, pobs, gamma, v3, gamma2, u1, u3):
    xobs = robs * np.cos(pobs) * np.sin(tobs)
    yobs = robs * np.sin(pobs) * np.sin(tobs)
    zobs = robs * np.cos(tobs)

    xem = rem * np.cos(pem) * np.sin(tem)
    yem = rem * np.sin(pem) * np.sin(tem)
    zem = rem * np.cos(tem)

    S = xobs * xem + yobs * yem + zobs * zem

    f1 = gamma * gamma2 * (1 - v3 * u3) - gamma2 * u1 * (rem**2 - S) / rem
    f2 = robs * np.sin(tem) * np.sin(pobs - pem) * (gamma * v3 - gamma2 * u3 * (1 + gamma**2 * v3**2 / (1 + gamma)))

    print(f1, gamma)
    print(np.sin(tem) * np.sin(pobs - pem), rem * np.sin(tem) * dphi(0., rem, tem, pem, robs, tobs, pobs) )
    print(-4.494169261164169 / (rem * np.sin(tem)))
    print((rem**2 - S) / (rem * robs))

    dr = - (rem**2 - S) / (robs * rem)
    print(dr)
    alpha = np.sin(tem) * np.sin(pobs - pem) * robs / dr

    print(5.41777546597162 / alpha)
    print(0.05922595578847367 / (robs * np.cos(tobs) - S * np.cos(tem)) / (robs * np.sin(tem)))
    return 1 / (f1 - f2 / (robs))


def main():
    rem = 7.99655900302855
    tem = 1.5708846219961554
    pem = -2.2753562911782295 #+ np.pi * 2

    robs = 35.
    pobs = 0.
    tobs = 1.

    v3 = 0.5
    gamma = 1.1547005383792517
    u1 = 0.
    u3 = 0.
    gamma2 = 1.

    print(red(rem, tem, pem, robs, tobs, pobs, gamma, v3, gamma2, u1, u3))


if __name__ == '__main__':
    main()
