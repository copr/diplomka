import numpy as np
import numpy.random as dist
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as st
from mcmc import metropolis_hastings as mh


def get_stat_ig(data, mu, a, b):
    an = a + len(data)/2
    bn = b + 0.5*np.sum((data - mu)**2)
    print(an, bn)
    g = st.gamma(a = an, loc=0, scale = 1/bn)
    def comp(sig):
        nonlocal g
        return g.pdf(sig)
    return comp



if __name__ == "__main__":
    mu = 0
    precision = 0.1111
    n = 1000
    data = dist.normal(mu, (1/precision)**0.5, n)

    a = 2
    b = 1
    
    stat = get_stat_ig(data, mu, a, b)

    sd = 0.1
    prop = lambda x, xi: norm(xi, sd).pdf(x)
    prop_sampler = lambda x: dist.normal(x, sd)


    samples = mh(stat, prop, prop_sampler, 2000)

    print(np.mean(samples))
