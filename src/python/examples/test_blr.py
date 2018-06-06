import numpy as np
import numpy.linalg as npla
import numpy.random as dist
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import multivariate_normal as normal
from scipy.stats import invgamma, gamma
from mcmc import metropolis_hastings_kth as mh
from mpl_toolkits.mplot3d import Axes3D as ax



T = lambda x: x.transpose()

A0 = np.array([[1, 0.5], [0.5, 1]])
mu0 = np.array([[5], [0]])
a0 = 1
b0 = 10

data = np.loadtxt('../resources/data/discoveries.csv', delimiter=',', usecols=(1,2), skiprows=1)


dataX = np.arange(1, 10.5, 1)
dataY = np.array([1,1,1,1,1,6,7,8,9,10])# + np.random.normal(0, 2, 10) + 5

# dataX = data[:, 0]
# dataY = data[:, 1]

X = T(np.array([np.ones(len(dataX)), dataX]))
Y = T(np.array([dataY]))

n = len(dataX)

BhatI = npla.inv(np.dot(T(X),X))
BhatR = np.dot(T(X), Y)
Bhat = np.dot(BhatI, BhatR)

An = np.dot(T(X), X) + A0
Ainv = npla.inv(An)
munleft = npla.inv(An)
munright = np.dot(A0, mu0) + np.dot(T(X), np.dot(X, Bhat))
mun = np.dot(munleft, munright)
bn = b0 + 0.5*(np.dot(T(Y), Y) + np.dot(T(mu0), np.dot(A0, mu0)) - np.dot(T(mun), np.dot(An, mun)))
an = a0 + n/2


mean = mun[:,0]


q = bn[0][0]

stat = lambda B: normal(mean, abs(B[2])*Ainv).pdf(B[0:2]) * invgamma(a=an, loc=0, scale=q).pdf(B[2])


k1 = 0.05
k2 = 0.05
k3 = 5
cov_sampler = np.array([[k1, 0, 0], [0, k2, 0], [0, 0, k3]])

prop = lambda x, xi: normal(xi, cov_sampler).pdf(x)
prop_sampler = lambda x: normal(x, cov_sampler).rvs()

cov_sampler2 = np.array([[20, 0.5, 0], [0.5, 20, 0.5], [0, 0.5, 20]])
                 
prop2 = lambda x, xi: normal(xi, cov_sampler2).pdf(x)
prop_sampler2 = lambda x: normal(x, cov_sampler2).rvs()

k = 5

samples, failures = mh(stat, prop, prop_sampler, k, prop, prop_sampler,
                       2000, np.array([5, 1, 10]))


b0 = np.mean(samples[:, 0])
b1 = np.mean(samples[:, 1])
b0confidence_95 = np.percentile(samples[:, 0], [2.5, 97.5])
b1confidence_95 = np.percentile(samples[:, 1], [2.5, 97.5])

mindataX = np.min(dataX)
maxdataX = np.max(dataX)
      
plt.figure(1)
plt.subplot(221)
plt.plot(dataX, Y, 'ro')
plt.plot([mindataX, maxdataX], [b0 + mindataX*b1,
                                b0 + maxdataX*b1])
plt.plot([mindataX, maxdataX], [b0confidence_95[0] + mindataX*b1confidence_95[0],
                   b0confidence_95[0] + maxdataX*b1confidence_95[0]], 'g')
plt.plot([mindataX, maxdataX], [b0confidence_95[1] + mindataX*b1confidence_95[1],
                   b0confidence_95[1] + maxdataX*b1confidence_95[1]], 'g')


plt.subplot(222)
plt.plot(range(len(samples)), samples[:, 0])
plt.plot(range(len(samples)), failures[:, 0], 'rx')
plt.title('b0')
plt.subplot(223)
plt.plot(range(len(samples)), samples[:, 1])
plt.plot(range(len(samples)), failures[:, 1], 'rx')
plt.title('b1')

plt.subplot(224)
plt.plot(range(len(samples)), samples[:, 2])
plt.plot(range(len(samples)), failures[:, 2], 'rx')
plt.title('sigma2')

plt.show()


