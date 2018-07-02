import pickle
import numpy as np
import matplotlib.pyplot as plt


import utils

from revers import Rjmcmc
from blrMoveFactory import BlrMoveFactory
from stationaryDistributionFactory import StationaryDistributionFactory
from mcmcFactory import McmcFactory
from plotter import Plotter

np.random.seed(0)

#xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# ys = np.array([0, 0, 0, 0, 0, 1, 2, 3, 3, 4, 4]) + np.random.normal(0, 1, 11)
#ys = np.array([12, 10, 8, 6, 4, 2, 4, 6, 8, 10, 12]) + np.random.normal(0, 0.1, 11)
#ys = np.array([20, 15, 7, 1, 1, 1, 1, 11, 18, 25, 32]) + np.random.normal(0, 2, len(xs))

# xs = np.array([0, 2, 4, 6])
# origy = np.array([5, 0, 0, -5])
# ys = origy  + np.random.normal(0, 0.25, len(xs))

# xs = np.array([0, 1, 2, 3, 4, 5, 6])
# origy = np.array([5, 2.5, 0, 0, 0, -2.5, -5])
# ys = origy + np.random.normal(0, 0.25, len(xs))

line_points_x = [0, 3, 6, 7, 10]
line_points_y = [10, 5, 5, 3, 3]

var = 1
n = 10

xs, ys = utils.generate_samples(line_points_x, line_points_y, n, var)

stats = StationaryDistributionFactory(xs, ys)
moves = BlrMoveFactory()
mcmcs = McmcFactory(stats)

#first_sample = [1, 0, 0, 3, -5, 6, 5, 7, 3, 10, 5]


first_sample = stats.get_stationary(3).generate_first_sample()

rjmcmc = Rjmcmc(moves, mcmcs, stats)

samples = rjmcmc.sample(20000, (3, first_sample))

plotter = Plotter(xs, ys, line_points_x, line_points_y, samples)

file_name = '_'.join('samples/' + str(hash(plotter)),
                     str(len(line_points_x)),
                     str(n),
                     str(var))

with open(file_name, 'wb') as f:
    pickle.dump(plotter, f)
