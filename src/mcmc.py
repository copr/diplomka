import sys
import numpy as np
import numpy.random as dist
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as st



def metropolis_hastings(stat, prop, prop_sampler, n, first_sample=0):
# metropolis-hastings algoritmus pro ziskani "n" vzorku ze stacionardni distribuce "stat"
# pomoci navrhovaci distribuce "prop", z ktere se navrhuje pomoci "prop_sampler"
    samples = np.zeros(n)
    samples[0] = first_sample
    for i in range(1, n):
        u = dist.uniform()
        xprev = samples[i-1]
        xprop = prop_sampler(xprev)
        up = stat(xprop)*prop(xprev, xprop)
        down = stat(xprev)*prop(xprop, xprev)
        if u < up/down:
            samples[i] = xprop
        else:
            samples[i] = xprev
    return samples

def metropolis_hastings_kth(stat, prop, prop_sampler, k, prop_kth, prop_sampler_kth, n, first_sample=0):
    '''
    metropolis-hastings algoritmus pro vzorkovani z dane stacionardni distribuce
    
    stat - stacionarni distribuce z ktere se bude vzorkovat
    prop - funkce, ktera spocita pravdepodobnost vzorku z navrhove distribuce 
    prop_sampler - funkce, ktera navrhne dalsi vzorek z navrhove distribuce
    k - pro kazdy kty vzorek se pouzije jina distribuce
    prop_kth - obdobne jako predtim akorat pro kty vzorek
    prop_sampler_kth - obdobne jako predtim akorat pro kty vzorek
    n - pocet vzorku, ktere se vygenerujou
    first_sample - urcuje jak bude vypadat prvni vzorek

    pouziti: # v tomto pripade se navrhuje vzdycky ze stejne distribuce pro kty prvek se nic nemeni
    cov = np.matrix([[1, 0], [0, 10]])
    mean = np.array([0, 0])
    N_stat = norm(mean, cov)
    stat = lambda x: N_stat.pdf(x)

    cov_sampler = np.matrix([[3, 0], [0, 3]])

    prop = lambda x, xi: norm(xi, cov_sampler).pdf(x)
    prop_sampler = lambda x: norm(x, cov_sampler).rvs()
    k = 5

    samples = mh(stat, prop, prop_sampler, k, prop, prop_sampler, 2000, np.array([0, 0]))
    '''
    samples = np.empty([n, len(first_sample)])
#    samples = np.zeros(n)
    samples[0] = first_sample
    sys.stdout.write("\tSampling: \n")
    sys.stdout.flush()
    for i in range(1, n):
        u = dist.uniform()
        xprev = samples[i-1]
        if i % k == 0:
            xprop = prop_sampler_kth(xprev)
            up = stat(xprop)*prop_kth(xprev, xprop)
            down = stat(xprev)*prop_kth(xprop, xprev)
        else:
            xprop = prop_sampler(xprev)
            up = stat(xprop)*prop(xprev, xprop)
            down = stat(xprev)*prop(xprop, xprev)
        if u < up/down:
            samples[i] = xprop
        else:
            samples[i] = xprev
        # progress bar
        sys.stdout.write("\r\t%.0f%% Done" % (100*i/n))
        sys.stdout.flush()
    # progress konec
    sys.stdout.write("\r\t100% Done")
    sys.stdout.flush()
    return samples

if __name__ == "__main__":
    stat = lambda x: (5*(x > -5 and x < 0) + (5+x**3)*(x > 0 and x < 5))/206.25#0.3*np.exp(-0.2*x**2) + 0.7*np.exp(-0.2*(x-10)**2)
    prop = lambda xstar, xi: norm(xi, 100).pdf(xstar)
    prop_sampler  = lambda x: dist.normal(x, 100)

    samples = metropolis_hastings(stat, prop, prop_sampler, 5000)

    plt.hist(samples, normed=True, bins=20)
    xs = np.linspace(np.min(samples), np.max(samples), 100)
    statxs = np.zeros(len(xs))
    for i in range(len(xs)):
        statxs[i] = stat(xs[i])
    plt.plot(xs, statxs, 'r-')
    plt.show()
