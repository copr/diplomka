import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as normal
from scipy.stats import uniform

import utils

from revers import Rjmcmc
from blr2 import Blr2 as Blr
from blrMoveFactory import BlrMoveFactory
from stationaryDistributionFactory import StationaryDistributionFactory
from mcmcFactory import McmcFactory

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

xs = np.array([0, 0.5, 1, 2, 3, 3.5, 4, 5, 5.5, 6])
origy = np.array([5, 3.75, 2.5, 0, 0,  0, 0, -2.5, -3.75, -5])
ys = origy + np.random.normal(0, 0.25, len(xs))


stats = StationaryDistributionFactory(xs, ys)
moves = BlrMoveFactory()
mcmcs = McmcFactory(stats)


first_sample = stats.get_stationary(0).generate_first_sample()
# first_sample = np.array([1, 0, 0, 5, 0, 10, 12])


rjmcmc = Rjmcmc(moves, mcmcs, stats)

samples = rjmcmc.sample(20000, (0, first_sample))
# samples = samples[10000:]

model1_samples = [x[1] for x in samples if x[0] == 1]
model2_samples = [x[1] for x in samples if x[0] == 2]
model3_samples = [x[1] for x in samples if x[0] == 3]

ks = [x[0] for x in samples]
n1 = len(model1_samples)
n2 = len(model2_samples)
n3 = len(model3_samples)

ss1, hs1, sigmas1 = utils.individual_samples(model1_samples, blr0.n)
ss2, hs2, sigmas2 = utils.individual_samples(model2_samples, blr1.n)

s1means = np.mean(ss1, 1)
h1means = np.mean(hs1, 1)

s2means = np.mean(ss2, 1)
h2means = np.mean(hs2, 1)

# plt.plot(xs, ys, '*')
# plt.plot(s2means, h2means)
# plt.show()

print(n1, n2, n3)
utils.plot_lines(xs, ys, model3_samples, blr2.n, 50, 0.1, origy)
utils.plot_hists(xs, ys, model3_samples, blr2.n, 50, 0.1, origy)
