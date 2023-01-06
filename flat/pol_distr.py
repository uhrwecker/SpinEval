import os
import time

from new_pol_this_time_its_me import evaluate


def main():
    s = -0.001
    rem = 8.0

    fp_to_data = '/media/jan-menno/T7/Flat/v05/s0/'
    fp_to_save = '/media/jan-menno/T7/Flat/pol_at_obs/s-01/'

    print('Load data (this may take some time) ...')
    start = time.time()

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

        evaluate(s, rem, file, fp_to_save + phi + '/')

    t = time.time() - start
    print(f'Done! Took {t} s (or {t / 60} min).')

if __name__ == '__main__':
    main()