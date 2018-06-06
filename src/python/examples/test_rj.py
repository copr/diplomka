from rjmcmc.rj import test
import numpy as np
import numpy.random as dist
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as norm
import scipy.stats as st

sigma = 1
stat1 = lambda x: norm(0, sigma).pdf(x)


samples, pokusy = test(3000)

model1 = [x for x in samples if x[0] == 0]
data1 = [x[1] for x in model1]
model2 = [x for x in samples if x[0] == 1]
data2 = [(x[1],x[2]) for x in model2]

mu1 = np.mean(data1)
mu2 = np.mean(data2, 0)

print(mu1)
print(mu2)

points = np.linspace(-10, 10, 1000)

plt.plot(points, st.norm(mu1, sigma).pdf(points), 'r')
# plt.plot(points, st.norm(mu2[0], sigma).pdf(points), 'b')
# plt.plot(points, st.norm(mu2[1], sigma).pdf(points), 'b')
#plt.hist(data)
plt.show()
