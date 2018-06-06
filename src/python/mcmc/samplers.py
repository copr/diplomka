import sys
import numpy as np
import numpy.random as dist
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal as normal
import scipy.stats as st
import warnings
import copy

def super_metropolis_hastings(stat, proposal, proposal_sampler, n, first_sample):
    '''
    Zatim fungoval nejlip tak ho budu pouzivat.

    proposal(x, y) = proposal(y).pdf(x)
    proposal_sampler(x) = proposal(x).rvs()

    metropolis-hastings algoritmus pro ziskani "n"
    vzorku ze stacionardni distribuce "stat"
    pomoci navrhovaci distribuce "prop", z ktere
    se navrhuje pomoci "prop_sampler"
    '''
    samples = np.zeros([n, len(first_sample)])
    samples[0] = first_sample
    successess = 0
    nn = 0
    for i in range(1, n):
        xprev = samples[i-1]
        xprop = proposal_sampler(xprev)

        local_xprev = copy.copy(xprev)
        for j, x in enumerate(xprop):
            local_xprev[j] = x
            down = stat(xprev)*proposal(local_xprev, xprev)
            up = stat(local_xprev)*proposal(xprev, local_xprev)
            u = dist.uniform()
            nn += 1
            successess += 1
            if u > up/down:
                local_xprev[j] = xprev[j]
                successess -= 1

        samples[i] = local_xprev

        # progress bar
        sys.stdout.write("\r\t%.0f%% Done" % (100*i/n))
        sys.stdout.flush()
    # progress konec
    sys.stdout.write("\r\t100% Done\n")
    sys.stdout.flush()
    return (samples, successess/nn)

def metropolis_hastings(stat, prop, prop_sampler, n, first_sample=0):
# metropolis-hastings algoritmus pro ziskani "n" vzorku ze stacionardni distribuce "stat"
# pomoci navrhovaci distribuce "prop", z ktere se navrhuje pomoci "prop_sampler"
    samples = [first_sample]
    successess = 0
    for i in range(1, n):
        u = dist.uniform()
        xprev = samples[i-1]
        xprop = prop_sampler(xprev)
        up = stat(xprop)*prop(xprev, xprop)
        down = stat(xprev)*prop(xprop, xprev)
        if u < up/down:
            successess += 1
            samples.append(xprop)
        else:
            samples.append(xprev)
        # progress bar
        sys.stdout.write("\r\t%.0f%% Done" % (100*i/n))
        sys.stdout.flush()
    # progress konec
    sys.stdout.write("\r\t100% Done\n")
    sys.stdout.flush()
    return (samples, successess/n)

def special_mean(xs):
    suma = np.zeros(len(xs[0]))
    for x in xs:
        xx = np.array(x)
        suma += xx
    return suma/(len(xs)+1)


def covariance(xs):
    mean = np.mean(xs, 0)
    suma = 0
    k = len(xs)
    for x in xs:
        xx = np.array(x)
        suma += np.outer(xx, xx) - np.outer(mean, mean)
    return suma/k


def adaptive_metropolis1(stat, n, first_sample, cov, t0):
# metropolis-hastings algoritmus pro ziskani "n" vzorku ze stacionardni distribuce "stat"
# pomoci navrhovaci distribuce "prop", z ktere se navrhuje pomoci "prop_sampler"
    samples = [first_sample]
    successess = 0
    d = len(first_sample)
    sd = 2.4**2/d
    eps = 0.1
    xs = np.array(first_sample, dtype='float64')
    nn = 0
    for i in range(1, n):
        u = dist.uniform()
        xprev = samples[i-1]
        xprop = normal(xprev, cov).rvs()
        up = stat(xprop)
        down = stat(xprev)
        local_xprev = copy.copy(xprev)
        for j, x in enumerate(xprop):
            local_xprev[j] = x
            up = stat(local_xprev)
            u = dist.uniform()
            nn += 1
            successess += 1
            if u > up/down:
                local_xprev[j] = xprev[j]
                successess -= 1
        if u < up/down:
            successess += 1
            samples.append(xprop)
        else:
            samples.append(xprev)
        xs += np.array(samples[i])
        if i == t0:
            cov = sd*covariance(samples) + sd*eps*np.eye(d)
        elif i > t0:
            sample = np.array(samples[i])
            xprev_mean = (xs-sample)/i
            xprop_mean = xs/(i+1)
            xprev_squared = np.outer(xprev_mean, xprev_mean)
            xprop_squared = np.outer(xprop_mean, xprop_mean)
            newstx_squared = np.outer(sample, sample)
            cov = (i-1)*cov/i + sd*(i*xprev_squared - (i+1)*xprop_squared + newstx_squared + eps*np.eye(d))/i

        # progress bar
        sys.stdout.write("\r\t%.0f%% Done" % (100*i/n))
        sys.stdout.flush()
    # progress konec
    sys.stdout.write("\r\t100% Done\n")
    sys.stdout.flush()
    print(cov)
    return (samples, successess/nn)




def adaptive_metropolis_var(stat, n, first_sample, cov):
    '''
    metropolis-hastings algoritmus pro ziskani "n"
    vzorku ze stacionardni distribuce "stat"
    pomoci navrhovaci distribuce "prop", z ktere
    se navrhuje pomoci "prop_sampler"
    '''
    samples = [first_sample]
    successess = 0
    nn = 0
    for i in range(1, n):
        xprev = samples[i-1]
        xprop = normal(xprev, cov).rvs()
        down = stat(xprev)
        local_xprev = copy.copy(xprev)
        for j, x in enumerate(xprop):
            local_xprev[j] = x
            up = stat(local_xprev)
            u = dist.uniform()
            nn += 1
            successess += 1
            if u > up/down:
                local_xprev[j] = xprev[j]
                successess -= 1

        samples.append(local_xprev)

        # progress bar
        sys.stdout.write("\r\t%.0f%% Done" % (100*i/n))
        sys.stdout.flush()
    # progress konec
    sys.stdout.write("\r\t100% Done\n")
    sys.stdout.flush()
    return (samples, successess/nn)

def adaptive_metropolis_var3(stat, n, first_sample, cov):
    '''
    metropolis-hastings algoritmus pro ziskani "n"
    vzorku ze stacionardni distribuce "stat"
    pomoci navrhovaci distribuce "prop", z ktere
    se navrhuje pomoci "prop_sampler"
    '''
    samples = [first_sample]
    successess = 0
    nn = 0
    for i in range(1, n):
        xprev = samples[i-1]
        xprop = normal(xprev, cov).rvs()
        down = stat(xprev)
        local_xprev = copy.copy(xprev)
        for j, x in enumerate(xprop):
            local_xprev[j] = x
            up = stat(local_xprev)
            u = dist.uniform()
            nn += 1
            successess += 1
            if u > up/down:
                local_xprev[j] = xprev[j]
                successess -= 1
                if abs(cov[j,j]) < 10:
                    cov[j, j] = cov[j,j]*1.01
            else:
                if abs(cov[j,j]) > 0.2:
                    cov[j, j] = cov[j,j]*0.99
        samples.append(local_xprev)

        # progress bar
        sys.stdout.write("\r\t%.0f%% Done" % (100*i/n))
        sys.stdout.flush()
    # progress konec
    sys.stdout.write("\r\t100% Done\n")
    sys.stdout.flush()
    print(cov)
    return (samples, successess/nn)


def adaptive_metropolis_var2(stat, n, first_sample, cov):
    '''
    metropolis-hastings algoritmus pro ziskani "n"
    vzorku ze stacionardni distribuce "stat"
    pomoci navrhovaci distribuce "prop", z ktere
    se navrhuje pomoci "prop_sampler"
    '''
    samples = [first_sample]
    successess = 0
    nn = 0
    for i in range(1, n):
        xprev = samples[i-1]
        xprop = normal(xprev, cov).rvs()
        down = stat(xprev)
        new_xprev = copy.copy(xprev)
        for j, x in enumerate(xprop):
            local_xprev = copy.copy(xprev)
            local_xprev[j] = x
            up = stat(local_xprev)
            u = dist.uniform()
            nn += 1
            if u < up/down:
                new_xprev[j] = x
                successess += 1
        samples.append(new_xprev)
        # progress bar
        sys.stdout.write("\r\t%.0f%% Done" % (100*i/n))
        sys.stdout.flush()
    # progress konec
    sys.stdout.write("\r\t100% Done\n")
    sys.stdout.flush()
    return (samples, successess/nn)



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
    samples = [first_sample]
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
        b = 1
        b = up/down
        if u < b:
            xprev = xprop
            samples.append(xprop)
            # failures.append(np.nan)
        else:
            xprev = xprev
            samples.append(xprev)
            # failures.append(xprop)# if i % 2 == 0 else np.nan
            # progress bar
        sys.stdout.write("\r\t%.0f%% Done" % (100*i/n))
        sys.stdout.flush()
    # progress konec
    sys.stdout.write("\r\t100% Done\n")
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
