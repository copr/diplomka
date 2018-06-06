import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as normal
from functools import partial

import sys
sys.path.append("/home/copr/git/diplomka/src/python/mcmc")

from samplers import metropolis_hastings as mh
from samplers import metropolis_hastings_kth as mh_kth
from samplers import adaptive_metropolis1 as am
from samplers import adaptive_metropolis_var as amv


class Blr_nbreaks:
    def __init__(self, xs, ys, n_breaks, prior_mu, prior_cov):
        self.xs = xs
        self.ys = ys
        self.n_breaks = n_breaks
        self.prior_mu = prior_mu
        self.prior_cov = prior_cov
        self.dimension = 3+2*n_breaks

        def blr_nbreaks(xs, ys):
            '''
            vraci funkci co spocita posterior hustotu pro bayes linear regresi
            s n zlomy
            ys, xs - ys pozorovane hodnoty, xs nezavisle hodnoty
            b0 - y=b00*x+b0
            sigma - rozpytyl
            b1s - n+1 sklonu primky
            switches - n zlomu primky
            '''
            # prvni budde vzdycky b0 a sigma pak b1s a swithces
            # takze delka jednoho samplu by mela byt vzdycky
            # 1 + 1 + (n+1) + n = 3+2n
            prior = partial(normal(prior_mu, prior_cov).pdf)
            n_breaks = self.n_breaks

            def prob_density(x):
                assert len(x) == self.dimension
                
                b0 = x[0]
                sigma = x[1]
                b1s = x[2:3+n_breaks]
                switches = x[3+n_breaks:]
                # pridam si nekonecno do switchu at muzu tesstovat
                # jen z jedne strany
                switches = np.append(switches, np.inf)
                prob = 0
                n = len(xs)
                for i, xi in enumerate(xs):
                    # v jake hodnote zacinam
                    for j, switch in enumerate(switches):
                        if j == 0:
                            a = b0
                        else:
                            a = a + b1s[j-1]*switches[j-1] - b1s[j]*switches[j-1]
                        if xi < switch:
                            prob += (ys[i] - (xi*b1s[j]+a))**2
                            break

                sigma = abs(sigma)
                if sigma < 0:
                    raise Exception("sigma < 0")
                prob = (sigma)**(-n/2) * np.exp(-prob/(2*sigma))
                return prob*prior(x)
            return prob_density

        self.stationary = blr_nbreaks(xs, ys)

    def sample_am(self, n):
        first_sample = np.ones(self.dimension)
        # sigma
        samples, a = am(self.stationary,
                        n,
                        first_sample,
                        np.eye(self.dimension)*2,
                        1)
        self.samples = samples
        return samples, a

    def sample_am_var(self, n):
        first_sample = np.ones(self.dimension)
        # sigma
        samples, a = amv(self.stationary,
                         n,
                         first_sample,
                         np.eye(self.dimension)*2)
        self.samples = samples
        return samples, a

    def autocorr(self, x):
        return np.correlate(x, x, mode='full')[len(x)-1:]

    def draw_autocorealtion(self, samples):
        b00s = [x[0] for x in samples]
        b10s = [x[1] for x in samples]
        b0s = [x[2] for x in samples]
        # sigmas = [x[3] for x in samples]
        switches = [x[4] for x in samples]

        plt.figure(1)
        plt.subplot(221)
        plt.plot(range(len(samples)), self.autocorr(b00s))
        plt.title('autocorelation b00')

        plt.subplot(222)
        plt.plot(range(len(samples)), self.autocorr(b10s))
        plt.title('autocorelation b10')

        plt.subplot(223)
        plt.plot(range(len(samples)), self.autocorr(b0s))
        plt.title('autocorelation b0')

        plt.subplot(224)
        plt.plot(range(len(samples)), self.autocorr(switches))
        plt.title('autocorelation b1')
        plt.show()

    def draw_timeseries(self, samples):

        means = np.mean(samples, 0)
        sigmas = [x[1] for x in samples]
        b0s = [x[0] for x in samples]
        b1s = [x[2] for x in samples]
        b2s = [x[3] for x in samples]
        b3s = [x[4] for x in samples]
        switches = [x for x in samples[3+n_breaks:]]
        
        
        plt.figure(1)
        plt.subplot(221)
        plt.plot(range(len(samples)), b0s)
#        plt.plot([1, len(samples)], [b00, b00], 'r')
        plt.title('b0s')

        plt.subplot(222)
        plt.plot(range(len(samples)), b1s)
#        plt.plot([1, len(samples)], [b10, b10], 'r')
        plt.title('b1s')

        plt.subplot(223)
        plt.plot(range(len(samples)), b2s)
#        plt.plot([1, len(samples)], [b0, b0], 'r')
        plt.title('b2s')

        plt.subplot(224)
        plt.plot(range(len(samples)), sigmas)
#        plt.plot([1, len(samples)], [switch, switch], 'r')
        plt.title('sigmas')
        plt.show()

    def draw_blr(self, samples):
        xs = self.xs
        ys = self.ys

        # b0, sigmas, b10, ..., b1(n_breaks-1), switch1, ... switch(n_breaks)
        means = np.mean(samples, 0)
        b0 = means[0]
        sigma = means[1]
        b1s = means[2:3+n_breaks]
        switches = means[3+n_breaks:]

        a = b0
        plt.plot(xs, ys, 'ro')
        plt.plot([xs[0], switches[0]],
                 [a + xs[0]*b1s[0], a + switches[0]*b1s[0]])
        for i in range(1, len(b1s)-1):
            s = switches[i-1]
            a = a + b1s[i-1]*s - b1s[i]*s
            # o kolik posunout x do nuly
            # v jake hodnote zacinam
            sp = switches[i-1]
            sn = switches[i]
            plt.plot([sp, sn],
                     [a + sp*b1s[i],  a + sn*b1s[i]])
            #   [(sp-move)*b1s[i-1]+start, (sn-move)*b1s[i]+start])
        plt.plot(xs, ys, 'ro')
        sl = switches[-1]
        a = a + b1s[-2]*switches[-1] - b1s[-1]*switches[-1]
        plt.plot([sl, xs[-1]],
                 [a + sl*b1s[-1], a + (xs[-1])*b1s[-1]])
        plt.show()
        
    def draw(self, samples):
        self.draw_blr(samples)
        self.draw_timeseries(samples)

        
# xs = list(range(30))
# ys = np.concatenate([np.linspace(0, 15, 20) + np.random.normal(0, 1, 20),
#                      np.random.normal(15, 1, 10)])

# xs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# ys = [1, 1, 1, 1, 1, 2, 3, 4, 5]

xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
ys = np.array([1, 1, 1, 1, 1, 2, 4, 6, 8, 10, 9, 8,
               7, 6, 5, 8, 10, 12, 14, 16, 18]) + np.random.normal(0, 2, 21)

n_breaks = 1

# prior_mu_2breaks = np.ones(3 + 2*n_breaks)
# prior_mu[1] = 4
# prior_mu[5] = 5
# prior_mu[6] = 10

# 0 b0, 1 sigma, 2 b10, 3 b11, 4 b12, 5 b13, 6 switch1, 7 switch2, 8 switch3
prior_mu = np.zeros(3 + 2*n_breaks)
# prior_mu[1] = 4
# prior_mu[6] = 5
# prior_mu[7] = 10
# prior_mu[8] = 15

# b0, sigma, b10, b11, b12, switch1, switch2

# prior_cov_2breaks = np.array([[5, 0, 0, 0, 0, 0, 0],
#                       [0, 3, 0, 0, 0, 0, 0],
#                       [0, 0, 3, 0, 0, 0, 0],
#                       [0, 0, 0, 3, 0, 0, 0],
#                       [0, 0, 0, 0, 2, 0, 0],
#                       [0, 0, 0, 0, 0, 5, 0],
#                       [0, 0, 0, 0, 0, 0, 5]])
prior_cov = np.eye(3+2*n_breaks)*30
# prior_cov[6, 6] = 3
# prior_cov[7, 7] = 3
# prior_cov[8, 8] = 3
# prior_cov[4, 4] = 10

blr = Blr_nbreaks(xs, ys, n_breaks, prior_mu, prior_cov)
samples, a = blr.sample_am(10000)
print(a)
blr.draw(samples)

# samples, a =  blr.sample(5000)
# blr.draw(samples)

# samples, a = blr.sample_am(5000)
# print(a)
# blr.draw(samples)

# n_breaks = 1

# prior_mu = np.zeros(3 + 2*n_breaks)
# prior_mu[1] = 4
# prior_mu[4] = 5

# # b0, sigma, b10, b11, switch1
# prior_mu = np.array([0, 4, 0, 1, 5])

# prior_cov = np.array([[5, 0, 0, 0, 0],
#                       [0, 3, 0, 0, 0],
#                       [0, 0, 3, 0, 0],
#                       [0, 0, 0, 3, 0],
#                       [0, 0, 0, 0, 2]])

# blr = Blr_nbreaks(xs, ys, n_breaks, prior_mu, prior_cov)
# # samples, a =  blr.sample(5000)
# # blr.draw(samples)

# samples, a = blr.sample_am(5000)
# print(a)
# blr.draw(samples)
