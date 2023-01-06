import os
from polarization_of_file import eval_spin_stuff


def main(fp_to_data, fp_to_save):
    s = 0
    rem = 8.0

    phis = [x[0] for x in os.walk(fp_to_data)]
    phis = [phi for phi in phis if not phi.endswith('data')]
    phis = [phi + '/' for phi in phis if not phi == fp_to_data]

    for n, file in enumerate(phis):
        phi = file[len(fp_to_data):-1]

        print(f'Now at {phi} ... ({n+1} / {len(phis)})')

        try:
            os.mkdir(fp_to_save + phi)
            os.mkdir(fp_to_save + phi + '/data')
        except FileExistsError:
            continue

        eval_spin_stuff(s, rem, file, fp_to_save + phi + '/')


if __name__ == '__main__':
    fp_to_data = '/media/jan-menno/T7/Flat/s0/'
    fp_to_save = '/media/jan-menno/T7/Flat/pol/'
    main(fp_to_data, fp_to_save)