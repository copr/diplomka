import numpy as np
import numpy.random as dist
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as st
from mcmc import metropolis_hastings_kth as mh

if __name__ == "__main__":
    n1 = norm(-50, 1)
    n2 = norm(50, 2)
    stat = lambda x: n1.pdf(x) + n2.pdf(x)

    sd1 = 3
    prop = lambda x, xi: norm(xi, sd1).pdf(x)
    prop_sampler = lambda x: dist.normal(x, sd1)

    k = 5
    sd2 = 100
    prop_kth = lambda x, xi: norm(xi, sd2).pdf(x)
    prop_sampler_kth = lambda x: dist.normal(x, sd2)

    samples = mh(stat, prop, prop_sampler, k, prop_kth, prop_sampler_kth, 2000, 50)

    
