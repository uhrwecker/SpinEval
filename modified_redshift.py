import numpy as np
import matplotlib.pyplot as pl

import matplotlib as mp

from util import g


def main():
    fp = '/home/jan-menno/Data/10_12_21/alt/0.0/'

    data = np.loadtxt(fp + '/redshift.csv', delimiter=',', skiprows=1)
