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

line_points_x = [0, 5, 10, 15]
line_points_y = [5, 0, 0, 5]

var = 3
n = 100

xs, ys = utils.generate_samples(line_points_x, line_points_y, n, var)

stats = StationaryDistributionFactory(xs, ys)
moves = BlrMoveFactory()
mcmcs = McmcFactory(stats)




first_sample = [1, 0, 5, 5, 0, 14, 5]

if not stats.get_stationary(1).pdf(first_sample) > 0:
    raise Exception("First sample has zero probality")

rjmcmc = Rjmcmc(moves, mcmcs, stats)

samples = rjmcmc.sample(20000, (1, first_sample))

plotter = Plotter(xs, ys, line_points_x, line_points_y, samples)

file_name = '_'.join(['samples/new_' + str(hash(plotter)),
                      str(len(line_points_x)),
                      str(n),
                      str(var)])

with open(file_name, 'wb') as f:
    pickle.dump(samples, f)
