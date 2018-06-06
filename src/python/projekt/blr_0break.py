import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, invgamma, expon
from scipy.stats import multivariate_normal as normal

import sys
sys.path.append("/home/copr/zdrojaky/diplomka/src/mcmc")

from mcmc import metropolis_hastings as mh
from mcmc import metropolis_hastings_kth as mh_kth


class Blr:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

        def blr_0break(xs, ys):
            '''
            vraci funkci co spocita posterior hustotu pro bayes linear regresi
            s jednim zlomem
            ys, xs - ys pozorovane hodnoty, xs nezavisle hodnoty
            b00,b10 - koeficient u zavisle promenne pres respektive za zlomem
            b0 - y=b00*x+b0
            sigma - rozpytyl
            s - zlomn
            '''
            prior_b1 = lambda x: uniform(loc=-10, scale=20).pdf(x)
            prior_b0 = lambda x: uniform(loc=-10, scale=20).pdf(x)
            prior_sigma = lambda x: uniform(1, scale=5).pdf(x)
            
            def prob_density(x):
                prob = 0
                n = len(xs)
                for i, xi in enumerate(xs):
                    prob += (ys[i] - (xi*x[0]+x[1]))**2
                prob = (x[2])**(-n/2) * np.exp(-prob/(2*abs(x[2])))
                return np.product([prob,
                                   prior_b0(x[0]),
                                   prior_b1(x[1]),
                                   prior_sigma(x[2])])
            return prob_density
        self.stationary = blr_0break(xs, ys)

    def proposal_sampler(self, x):
        '''
        returns new sample based on previou sample x
        '''
        cov = [[1, 0, 0],
               [0, 1, 0],
               [0, 0, 0.5]]
        return normal(x, cov).rvs()

    def proposal_distribution(self, xprev, xnext):
        '''
        returns probability of transtition from
        previous sample (xprev) to next smaple (xnext)
        '''
        cov = [[1, 0, 0],
               [0, 1, 0],
               [0, 0, 0.5]]
        return normal(xprev, cov).pdf(xnext)

    def sample(self, n):
        self.samples = mh(self.stationary, self.proposal_distribution,
                          self.proposal_sampler, n, [0, 0, 3])

    def draw(self):
        xs = self.xs
        ys = self.ys
        samples = self.samples
        
        b1s = [x[0] for x in samples]
        b0s = [x[1] for x in samples]
        sigmas = [x[2] for x in samples]

        b0 = np.mean(b0s)
        b1 = np.mean(b1s)
        sigma = np.mean(sigmas)

        plt.figure(1)
        plt.subplot(221)
        plt.plot(range(len(samples)), b0s)
        plt.title('b0s')

        plt.subplot(222)
        plt.plot(range(len(samples)), b1s)
        plt.title('b1s')

        plt.subplot(223)
        plt.plot(range(len(samples)), sigmas)
        plt.title('sigmas')

        plt.subplot(224)
        plt.plot(xs, ys, 'ro')
        plt.plot([min(xs), max(xs)], [b0 + min(xs)*b1, b0 + max(xs)*b1])
        plt.title('blr')
        plt.show()


xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ys = np.array([2, 3, 4, 5, 6, 7, 9, 10, 12, 11]) + np.random.normal(0, 0.1, 10)


blr = Blr(xs, ys)
blr.sample(2000)
blr.draw()
