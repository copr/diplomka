import numpy as np
import numpy.random as dist
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as norm
import scipy.stats as st
from mcmc import metropolis_hastings as mh


cov = np.matrix([[1, 0], [0, 10]])
mean = np.array([0, 0])
N_stat = norm(mean, cov)
stat = lambda x: N_stat.pdf(x)


cov_sampler = np.matrix([[3, 0], [0, 10]])

prop = lambda x, xi: norm(xi, cov_sampler).pdf(x)
prop_sampler = lambda x: norm(x, cov_sampler).rvs()

k = 5

samples = mh(stat, prop, prop_sampler, 10000, np.array([0, 0]))


mu0 = [x[0] for x in samples]
mu1 = [x[1] for x in samples]
plt.plot(mu0, mu1, 'ro')
axes = plt.gca()
axes.set_xlim([-15, 15])
axes.set_ylim([-15, 15])
plt.show()
