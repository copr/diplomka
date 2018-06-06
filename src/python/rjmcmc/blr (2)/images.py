import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as normal
from scipy.stats import uniform
from scipy.integrate import quad

from mcmc import Mcmc
from proposalDistribution import ProposalDistribution

# skript na vytvoreni ruznych obrazku do diplomky

xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ys = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5]) + np.random.normal(0, 0.1, 11)

plt.plot(xs, ys, 'bo')
plt.plot([0, 5.5, 10], [0, 0, 5], 'r')
plt.savefig('images/model_1break.png')

plt.gcf().clear()

plt.plot(xs, ys, 'bo')
plt.plot([0, 10], [-1, 4], 'r')
plt.savefig('images/model_0break.png')

plt.gcf().clear()

n1 = normal(0, 3)
n2 = normal(5, 1)

def stat(x):
    return (n1.pdf(x) + n2.pdf(x))


n3 = normal(0, 2)


def rvs(x):
    n3.mean = x
    return [n3.rvs()]


def pdf(x, y):
    n3.mean = x
    return n3.pdf(y)


stationary = ProposalDistribution(stat, stat)
proposal = ProposalDistribution(pdf, rvs)

normative, err = quad(stat, -10, 10)


mcmc = Mcmc(proposal, stationary)
samples = mcmc.sample(10000, [0])
samples = [sample[0] for sample in samples]

xs = np.linspace(-10, 10, 1000)
ys = [stat(x)/normative for x in xs]
plt.plot(xs, ys, 'r')
plt.hist(samples, normed=True, bins=100)
plt.savefig('images/mcmc_2normals.png')

plt.gcf().clear()
