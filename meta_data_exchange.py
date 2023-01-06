import copy

import numpy as np

def main():
    fp = '/media/jan-menno/T7/Flat/meta_data.txt'

    data = np.loadtxt(fp, delimiter=';')

    cop = copy.deepcopy(data)

    print(cop[-1])
    cop[:, 3] *= -1
    cop[:, 4] *= -1
    cop[:, [3, 4]] = cop[:, [4, 3]]
    cop[:, 2] = np.abs(cop[:, 2] - np.pi) + np.pi

    cop = cop[::-1]
    cop = np.delete(cop, -1, 0)

    data = np.append(data, cop, axis=0)
    print(data)

    np.savetxt(fp + '22', data, delimiter=';')



if __name__ == '__main__':
    main()