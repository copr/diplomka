import pymc as pc
import pymc.Matplot as pt
import numpy as np
from scipy.stats import bernoulli

def model(data):
    theta_prior = pc.Beta('theta_prior', alpha=1.0, beta=1.0)
    coin = pc.Bernoulli('coin', p=theta_prior, value=data, observed=True)
    mod = pc.Model([theta_prior, coin])

    return mod

def generateSample(t, s):
    return bernoulli.rvs(t, size=s)

def mcmcTraces(data):
    mod = model(data)
    mc = pc.MCMC(mod)

    mc.sample(iter=5000, burn=1000)
    return mc.trace('theta_prior')[:]

sample = generateSample(0.7, 500)

trs = mcmcTraces(sample)
print(trs)
k = pt.histogram(trs, "theta prior; size=100", datarange=(0.2, 0.9))
pt.plot(k, name='ahoj')
