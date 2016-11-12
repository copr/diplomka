import numpy as np
import numpy.linalg as npla
import numpy.random as dist
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import multivariate_normal as normal
from scipy.stats import invgamma
from mcmc import metropolis_hastings_kth as mh



T = lambda x: x.transpose()

A0 = np.array([[1, 0], [0, 1]])
mu0 = np.array([[0], [0]])
a0 = 1
b0 = 1

    
X = T(np.array([np.ones(10), [1,2,3,4,5,6,7,8,9,10]]))
Y = T(np.array([1,2,3,4,5,6,7,8,9,10]) + np.array([np.random.normal(0, 2, 10)])) + 5

n = 10

Bhat = np.dot(npla.inv(np.dot(T(X),X)), np.dot(T(X), Y))
An = np.dot(T(X), X) + A0
Ainv = npla.inv(An)
mun = np.dot(Ainv, np.dot(A0, mu0)) + np.dot(T(X), np.dot(X, Bhat))
bn = b0 + 0.5*np.dot(T(Y), Y) + np.dot(T(mu0), np.dot(A0, mu0)) - np.dot(T(mun), np.dot(An, mun))
an = a0 + n/2


mean = np.array(T(mun).tolist()[0])

q = bn[0][0]

stat = lambda B: normal(mean, np.abs(np.dot(B[2], Ainv))).pdf(B[0:2]) * invgamma(a=an, loc=0, scale=q).pdf(B[2])


cov_sampler = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                 
prop = lambda x, xi: normal(xi, cov_sampler).pdf(x)
prop_sampler = lambda x: normal(x, cov_sampler).rvs()

cov_sampler2 = np.array([[30, 0, 0], [0, 30, 0], [0, 0, 1]])
                 
prop2 = lambda x, xi: normal(xi, cov_sampler2).pdf(x)
prop_sampler2 = lambda x: normal(x, cov_sampler2).rvs()

k = 5

samples = mh(stat, prop, prop_sampler, k, prop2, prop_sampler2, 5000, np.array([1, 1, 1]))

b0 = np.mean(samples[:, 0])
b1 = np.mean(samples[:, 1])

plt.figure(1)
plt.plot([1,2,3,4,5,6,7,8,9,10], Y, 'ro')
plt.plot([1, 10], [b0 + 1*b1, b0 + 10*b1])

plt.figure(2)
plt.plot(range(len(samples)), samples[:, 0])

plt.figure(3)
plt.plot(range(len(samples)), samples[:, 1])

plt.figure(4)
plt.plot(range(len(samples)), samples[:, 2])

plt.show()
