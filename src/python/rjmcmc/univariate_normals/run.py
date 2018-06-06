import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as normal
from scipy.stats import invgamma
from scipy.stats import uniform
from functools import partial
from ipyparallel import Client

import rjmcmc

import theano.tensor as T
import theano

import sys
sys.path.append("/home/copr/git/diplomka/src/python/mcmc")

from samplers import super_metropolis_hastings as mh

'''
Test implementation of RJMCMC. There are only two models
onw has one gaussian and the other has two gaussians.
There are two datasets one generated from one gaussian
and the other from two gaussians.

So the parameters we are looking for are:
1.model (mu1, sigma1)
2.model (mu21, sigma21, mu22, sigma22)
where mus and sigmas are mean and variance respectively.

So the samples should look something like this
(k, mu1, sigma1) e.g. (1, 0, 1) 
or
(k, mu21, sigma21, mu22, sigma22) e.g. (2, -5, 3, 5, 3)

priors:
f(mu1) ~ normal(0, 10)
f(sigma1) ~ invgamma(1, 1)

f(mu21) ~ normal(-5, 5)
f(mu22) ~ normal(5, 5)
f(sigma21) ~ inggamma(1, 1)
f(sigma21) ~ invgamma(1, 1)

likelihood:
f(x | mu1, sigma1) ~ normal(mu1, sigma1)

f(x | mu21, sigma21, mu22, sigma22) ~
    normal(mu21, sigma21) * normal(mu22, sigma22)

posterior:
f(mu1, sigma1 | x) ~ f(x | mu1, sigma1)*f(mu1)*f(sigma1)

f(mu21, sigma21, mu22, sigma22 | x) ~ f(x | mu21, sigma21, mu22, sigma22)*
                                      f(mu21)*f(mu22)*f(sigma21)*f(sigma22)

where k stands for model number in this case it can be either 1 or 2.

transormations:

u ~ U(0, 4)
(mu1, sigma1) => (mu1 + u, sigma1, mu1 - u, sigma1)
(mu1, sigma1, mu2, sigma2) => ((mu1 + mu2)/2, (sigma1+sigma2)/2)


'''




def stationary1(dataset1):
    ''' 
    returns a function that will compute the value for 
    a sample given the parameters
    '''
    prior_mu1 = partial(normal(0, 10).pdf)
    prior_sigma1 = lambda x: normal(0, 10).pdf(x)*uniform(0, 10).pdf(x)

    n = normal()


    def likelihood1(x, mu1, sigma1):
        if sigma1 < 0:
            raise Exception("Variance can't be negative")
        
        n.mean = mu1
        n.cov = sigma1
        return n.pdf(x)

    def prob_density(sample):
        if not len(sample) == 2:
            raise Exception("Wrong sample length, actual sample length: " + str(len(sample)))
        
        mu1 = sample[0]
        sigma1 = sample[1]

        probabilities = np.zeros(len(dataset1))

        for i, x in enumerate(dataset1):
            probabilities[i] = likelihood1(x, mu1, sigma1)

        probability = np.prod(probabilities)

        return probability*prior_mu1(mu1)*prior_sigma1(sigma1)

    return prob_density




    
def getProposalSampler2():
    # n1 = normal(0, 3)
    # n2 = normal(0, 3)
    # n3 = normal(0, 3)
    # n4 = normal(0, 3)


    def proposalSampler(x):
        if not len(x) == 4:
            raise Exception("Wrong sample length")
        
        # n1.mean = x[0]
        # n2.mean = x[1]
        # n3.mean = x[2]
        # n4.mean = x[3]
        
        proposal = np.random.normal(x, [3, 3, 3, 3])
        proposal[1] = np.abs(proposal[1])
        proposal[3] = np.abs(proposal[3])

        return proposal

        # return [
        #     n1.rvs(),
        #     n2.rvs(),
        #     n3.rvs(),
        #     n4.rvs()]

    return proposalSampler


def getProposal2():
    # n1 = normal(0, 3)
    # n2 = normal(0, 3)
    # n3 = normal(0, 3)
    # n4 = normal(0, 3)

    cov = np.array([[3, 0, 0, 0],
                    [0, 3, 0, 0],
                    [0, 0, 3, 0],
                    [0, 0, 0, 3]])

    n = normal([0, 0, 0, 0], cov)

    def proposal(x, y):
        if not len(x) == 4:
            raise Exception("Wrong sample length")

        if not len(y) == 4:
            raise Exception("Wrong sample length")
        
        # n1.mean = x[0]
        # n2.mean = x[1]
        # n3.mean = x[2]
        # n4.mean = x[3]

        n.mean = x

        return n.pdf(y)

        # return np.prod([
        #     n1.pdf(y[0]),
        #     n2.pdf(y[1]),
        #     n3.pdf(y[2]),
        #     n4.pdf(y[3])
        # ])

    return proposal

def stationary2(dataset):
    ''' 
    returns a function that will compute the value for 
    a sample given the parameters
    '''

    n1 = normal(0, 10)
    n2 = normal(1, 2)
    u1 = uniform(-20, 20)
    u2 = uniform(0, 10)


    prior_mus = lambda mu1, mu2: n1.pdf(0) - n1.pdf(mu1-mu2)
    
#    prior_mu1 = lambda x: n1.pdf(x)
    prior_sigma1 = lambda x: u2.pdf(x)*n2.pdf(x)
#    prior_mu2 = lambda x: n1.pdf(x)
    prior_sigma2 = lambda x: u2.pdf(x)*n2.pdf(x)

    n1 = normal()
    n2 = normal()

    # c = Client()
    # v = c[:]

    def likelihood(x, mu1, sigma1, mu2, sigma2):
        sample = [mu1, sigma1, mu2, sigma2]
        if sigma1 < 0:
            raise Exception("Variance cannot be negative. Sample " + str(sample))
        
        if sigma2 < 0:
            raise Exception("Variance cannot be negative. Sample " + str(sample))
        
        n1.mean = mu1
        n1.cov = sigma1
        n2.mean = mu2
        n2.cov = sigma2
        return n1.pdf(x) + n2.pdf(x)

    def prob_density(sample):
        if not len(sample) == 4:
            raise Exception("Wrong sample length, actual sample length: " + str(len(sample)))
        
        mu1 = sample[0]
        sigma1 = sample[1]
        mu2 = sample[2]
        sigma2 = sample[3]

        probabilities = np.zeros(len(dataset))
        for (i, x) in enumerate(dataset):
            probabilities[i] = likelihood(x, mu1, sigma1, mu2, sigma2)
        probability = np.prod(probabilities)

        # probabilities = v.map(lambda x: likelihood(x, mu1, sigma1, mu2, sigma2), dataset)
        # probability = np.prod(probabilities.result())

        return np.prod([probability,
                        prior_mus(mu1, mu2),
                        prior_sigma1(sigma1),
                        prior_sigma2(sigma2)])

    return prob_density


x = T.vector('x')
y = T.vector('y')

hx = x[0] + x[2]
hy = x[1]*x[3]
hz = x[0] - x[2]
hw = x[1]/x[3]

h = T.as_tensor_variable([hx, hy, hz, hw])

hh = theano.function([x], h)

hj = theano.gradient.jacobian(h, x)

h_jacobian = theano.function([x], hj)

gx = (y[0] + y[2])/2
gy = (y[1] + y[3])/2
gz = y[2]
gw = y[3]

g = T.as_tensor_variable([gx, gy, gz, gw])

gg = theano.function([y], g)

gj = theano.gradient.jacobian(g, y)

g_jacobian = theano.function([y], gj)


def transformation1(x):
    transformed = hh(x)
    if x[1] < 0 or x[3] < 0:
        raise Exception("Wrong transformation")

    return transformed


def transformation2(x):
    transformed = gg(x)
    if x[1] < 0:
        raise Exception("Wrong transformation")

    return transformed


transformations = [transformation1, transformation2]
transformations_jacobians = [h_jacobian, g_jacobian]

#dataset2 = np.append(normal(-5, 1).rvs(5), [normal(5, 1).rvs(5)])
dataset2 = normal(10, 3).rvs(10)


dataset = dataset2

stat1 = stationary1(dataset)

def proposal(x, y):
    if not len(x) == 2 or not len(y) == 2:
        raise Exception("Wrong sample length")

    return normal(x[0], 3).pdf(y[0])*normal(x[1], 3).pdf(y[1])

def proposal_sampler(x):
    if not len(x) == 2:
        raise Exception("Wrong sample length")

    return [normal(x[0], 3).rvs(),
            np.abs(normal(x[1], 3).rvs())]


proposal2 = getProposal2()
proposal_sampler2 = getProposalSampler2()
stat2 = stationary2(dataset)

first_sample = np.array([1.0, 5.0, 3.0, 15, 3])
n = 10000

stationaries = [stat1, stat2]
proposals = [proposal, proposal2]
proposal_samplers = [proposal_sampler, proposal_sampler2]

start = time.time()
samples = rjmcmc.rjmcmc(stationaries, proposals, proposal_samplers,
                        transformations, transformations_jacobians,
                        first_sample, n)
end = time.time() - start

model0_samples = [w[1:] for w in samples if w[0] == 0]
model1_samples = [w[1:] for w in samples if w[0] == 1]

mus = [w[0] for w in model0_samples]
sigmas = [w[1] for w in model0_samples]

mus1 = [w[0] for w in model1_samples]
sigmas1 = [w[1] for w in model1_samples]
mus2 = [w[2] for w in model1_samples]
sigmas2 = [w[3] for w in model1_samples]

ewmus = np.mean(mus)
ewsigmas = np.mean(sigmas)

ewmus1 = np.mean(mus1)
ewmus2 = np.mean(mus2)
ewsigmas1 = np.mean(sigmas1)
ewsigmas2 = np.mean(sigmas2)


# mus1 = [w[0] for w in samples]
# sigmas1 = [w[1] for w in samples]
# mus2 = [w[2] for w in samples]
# sigmas2 = [w[3] for w in samples]

# ewMus1 = np.mean(mus1)
# ewSigmas1 = np.mean(sigmas1)
# ewMus2 = np.mean(mus2)
# ewSigmas2 = np.mean(sigmas2)

xs = np.linspace(-30, 30, 1000)

plt.hist(dataset2)
plt.plot(xs, normal(ewmus1, ewsigmas1).pdf(xs), 'r')
plt.plot(xs, normal(ewmus2, ewsigmas2).pdf(xs), 'r')
plt.plot(xs, normal(ewmus, ewsigmas).pdf(xs), 'b')
plt.show()
