import numpy as np
import numpy.linalg as npla
import numpy.random as dist
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import multivariate_normal as normal
from scipy.stats import invgamma
from mcmc import metropolis_hastings_kth as mh
from mpl_toolkits.mplot3d import Axes3D as ax



T = lambda x: x.transpose()

A0 = np.array([[1, 0], [0, 1]])
mu0 = np.array([[1800], [0]])
a0 = 1
b0 = 1

data = np.loadtxt('../resources/data/discoveries.csv', delimiter=',', usecols=(1,2), skiprows=1)


# dataX = np.arange(1, 10.5, 1)
# dataY = np.array([1,2,3,4,5,6,7,8,9,10]) + np.random.normal(0, 2, 10) + 5

dataX = data[:, 0]
dataY = data[:, 1]

X = T(np.array([np.ones(len(dataX)), dataX]))
Y = T(np.array([dataY]))

n = len(dataX)

Bhat = np.dot(npla.inv(np.dot(T(X),X)), np.dot(T(X), Y))
An = np.dot(T(X), X) + A0
Ainv = npla.inv(An)
munleft = npla.inv(An)
munright = np.dot(A0, mu0) + np.dot(T(X), np.dot(X, Bhat))
mun = np.dot(munleft, munright)
bn = b0 + 0.5*(np.dot(T(Y), Y) + np.dot(T(mu0), np.dot(A0, mu0)) - np.dot(T(mun), np.dot(An, mun)))
an = a0 + n/2


mean = np.array(T(mun).tolist()[0])

q = bn[0][0]

stat = lambda B: normal(mean, np.abs(np.dot(B[2], Ainv))).pdf(B[0:2]) * invgamma(a=an, loc=0, scale=q).pdf(B[2])


def fun(x, y):
  return normal(mean, np.abs(np.dot(800, Ainv))).pdf([x,y]) * invgamma(a=an, loc=0, scale=q).pdf(700)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(mun[0] - 300, mun[0] + 300, 5)
y = np.arange(mun[1] - 300, mun[0] + 300, 5)
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
