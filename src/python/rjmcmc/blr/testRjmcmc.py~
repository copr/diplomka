import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as normal
from scipy.stats import uniform

import utils

from move import Move
from revers import Rjmcmc
from mcmc import Mcmc
from mcmc_kth import Mcmc_kth
from blr2 import Blr2 as Blr
from proposalDistribution import ProposalDistribution

from transformations import hh, h_jacobian, gg, g_jacobian, ee, dd, e_jacobian, d_jacobian

xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# ys = np.array([0, 0, 0, 0, 0, 1, 2, 3, 3, 4, 4]) + np.random.normal(0, 1, 11)
#ys = np.array([12, 10, 8, 6, 4, 2, 4, 6, 8, 10, 12]) + np.random.normal(0, 0.1, 11)
ys = np.array([20, 15, 7, 1, 1, 1, 1, 11, 18, 25, 32]) + np.random.normal(0, 2, 11)

blr0 = Blr(xs, ys, 0)
blr1 = Blr(xs, ys, 1)
blr2 = Blr(xs, ys, 2)

n1 = normal(np.zeros(blr0.n), 5*np.eye(blr0.n))


def rvs0(theta):
    n1.mean = theta
    k = n1.rvs()
    k[0] = np.abs(k[0])
    return k


def pdf0(x, y):
    n1.mean = x
    return n1.pdf(y)


n2 = normal(np.zeros(blr1.n), 5*np.eye(blr1.n))


def rvs10(theta):
    n2.mean = theta
    k = n2.rvs()
    k[0] = np.abs(k[0])
    return k


def pdf10(x, y):
    n2.mean = x
    return n2.pdf(y)


n4 = normal(np.zeros(blr1.n), 5*np.eye(blr1.n))


def rvs11(theta):
    n4.mean = theta
    k = n4.rvs()
    k[0] = np.abs(k[0])
    return k


def pdf11(x, y):
    n4.mean = x
    return n4.pdf(y)


n8 = normal(np.zeros(blr2.n), 1*np.eye(blr2.n))


def rvs20(theta):
    n8.mean = theta
    k = n8.rvs()
    k[0] = np.abs(k[0])
    return k


def pdf20(x, y):
    n8.mean = x
    return n8.pdf(y)


n9 = normal(np.zeros(blr2.n), 5*np.eye(blr2.n))


def rvs21(theta):
    n9.mean = theta
    k = n9.rvs()
    k[0] = np.abs(k[0])
    return k


def pdf21(x, y):
    n9.mean = x
    return n9.pdf(y)


prop0 = ProposalDistribution(pdf0, rvs0)
prop10 = ProposalDistribution(pdf10, rvs10)
prop11 = ProposalDistribution(pdf11, rvs11)
prop20 = ProposalDistribution(pdf20, rvs20)
prop21 = ProposalDistribution(pdf21, rvs21)

mcmc0 = Mcmc(prop0, blr0)
mcmc1 = Mcmc_kth(prop10, prop11, blr1, 20)
mcmc2 = Mcmc_kth(prop20, prop21, blr2, 5)

first_sample = blr0.generate_first_sample()
# first_sample = np.array([1, 0, 0, 5, 0, 10, 12])

mcmcs = {1: mcmc0, 2: mcmc1, 3: mcmc2}

stationaries = {1: blr0, 2: blr1, 3: blr2}


def stationary(x):
    k, theta = x
    try:
        return stationaries[k].pdf(theta)
    except Exception as e:
        print(e)
        print(k)
        raise Exception("jebatz")

u = uniform(0, 1)
n3 = normal(0, 1)

uus_generated = []


def rvs_generator():
    uu = [u.rvs(), n3.rvs()]
    uus_generated.append(uu)
    return uu


def pdf_generator(x):
    return u.pdf(x[0])*n3.pdf(x[1])


generator = ProposalDistribution(pdf_generator, rvs_generator)

move1 = Move(1, 2, 0.05, 0.05, hh, gg,
            h_jacobian, g_jacobian,
            generator, None, 2, 0)

move2 = Move(2, 3, 0.05, 0.05, ee, dd,
             e_jacobian, d_jacobian,
             generator, None, 2, 0)

rjmcmc = Rjmcmc([move1, move2], mcmcs, stationary)

samples = rjmcmc.sample(20000, (1, first_sample))
# samples = samples[10000:]

model1_samples = [x[1] for x in samples if x[0] == 1]
model2_samples = [x[1] for x in samples if x[0] == 2]
model3_samples = [x[1] for x in samples if x[0] == 3]

ks = [x[0] for x in samples]
n1 = len(model1_samples)
n2 = len(model2_samples)
n3 = len(model3_samples)

us = [x[0] for x in uus_generated]
ns = [x[1] for x in uus_generated]

ss1, hs1, sigmas1 = utils.individual_samples(model1_samples, blr0.n)
ss2, hs2, sigmas2 = utils.individual_samples(model2_samples, blr1.n)

s1means = np.mean(ss1, 1)
h1means = np.mean(hs1, 1)

s2means = np.mean(ss2, 1)
h2means = np.mean(hs2, 1)

# plt.plot(xs, ys, '*')
# plt.plot(s2means, h2means)
# plt.show()

utils.plot_lines(xs, ys, model2_samples, blr1.n, 50, 0.1)
