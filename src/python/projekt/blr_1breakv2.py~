import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, invgamma, expon, chi
from scipy.stats import multivariate_normal as normal

import sys
sys.path.append("/home/copr/zdrojaky/diplomka/src")

from mcmc import metropolis_hastings as mh
from mcmc import metropolis_hastings_kth as mh_kth
from mcmc import adaptive_metropolis as am


class Blr_1break:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

        def blr_1break(xs, ys):
            '''
            vraci funkci co spocita posterior hustotu pro bayes linear regresi
            s jednim zlomem
            ys, xs - ys pozorovane hodnoty, xs nezavisle hodnoty
            b00,b10 - koeficient u zavisle promenne pres respektive za zlomem
            b0 - y=b00*x+b0
            sigma - rozpytyl
            s - zlomn
            '''
            prior_b00 = lambda x: normal(0, 3).pdf(x)
            prior_b10 = lambda x: normal(1, 3).pdf(x)
            prior_b0 = lambda x: normal(0, 3).pdf(x)
            prior_b1 = lambda x: normal(0, 5).pdf(x)
            prior_sigma = lambda x: normal(4, 3).pdf(x)
            prior_switch = lambda x: normal(15, 2).pdf(x)
            
            def prob_density(x):
                b00, b10, b0, b1, sigma, switch = x
                prob = 0
                n = len(xs)
                for i, xi in enumerate(xs):
                    if xi < switch:
                        prob += (ys[i] - (xi*b00+b0))**2
                    else:
                        prob += (ys[i] - (xi*b10+b1))**2
                sigma = abs(sigma)
                if sigma < 0:
                    raise Exception("sigma < 0")
                prob = (sigma)**(-n/2) * np.exp(-prob/(2*sigma))
                return np.product([prob,
                                   prior_b00(b00),
                                   prior_b10(b10),
                                   prior_b0(b0),
                                   prior_b1(b0),
                                   prior_sigma(sigma),
                                   prior_switch(switch)])
            return prob_density

        self.stationary = blr_1break(xs, ys)
        self.proposal_cov = np.array([[1, 0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 1]])
        self.proposal_cov_kth = np.array([[3, 0, 0, 0, 0, 0],
                                          [0, 3, 0, 0, 0, 0],
                                          [0, 0, 3, 0, 0, 0],
                                          [0, 0, 0, 3, 0, 0],
                                          [0, 0, 0, 0, 3, 0],
                                          [0, 0, 0, 0, 0, 3]])

    def proposal_sampler(self, x):
        '''
        returns new sample based on previou sample x
        '''
        bs = normal(x, self.proposal_cov).rvs()
        bs[4] = abs(bs[4])
        return bs
#        var = invgamma(a=1, scale=x[4]).rvs()
#        switch = normal(x[5], 2).rvs()
        
#        return np.concatenate([bs, [var], [switch]])

    def proposal_sampler_kth(self, x):
        '''
        returns new sample based on previou sample x
        '''
        bs = normal(x, self.proposal_cov_kth).rvs()
        bs[4] = abs(bs[4])
        return bs

    def proposal_distribution(self, xprev, xnext):
        '''
        returns probability of transtition from
        previous sample (xprev) to next smaple (xnext)
        '''
        bs = normal(xprev, self.proposal_cov).pdf(xnext)
        return bs
        # cov = np.array([[2, 0, 0, 0],
        #                 [0, 2, 0, 0],
        #                 [0, 0, 2, 0],
        #                 [0, 0, 0, 2]])
        # bs = normal(xprev[0:4], cov).pdf(xnext[0:4])
        # var = invgamma(a=1, scale=xprev[4]).pdf(xnext[4])
        # switch = normal(xprev[5], 2).pdf(xnext[5])
        # return bs*var*switch

    def proposal_distribution_kth(self, xprev, xnext):
        '''
        returns probability of transtition from
        previous sample (xprev) to next smaple (xnext)
        '''
        bs = normal(xprev, self.proposal_cov_kth).pdf(xnext)
        return bs

    def tune(self):
        cov = np.array([[1, 0, -0.5, 0, 0, 0.5],
                        [0, 1, 0, -0.5, 0, 0.5],
                        [-0.5, 0, 1, 0, 0, 0],
                        [0, -0.5, 0, 1, 0.1, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0.5, 0.5, 0, 0, 0, 1]])
        for i in range(1, 11):
            s, accept_rat = mh(self.stationary,
                               lambda xp, xn: normal(xp, cov*i).pdf(xn),
                               lambda x: normal(x, cov*i).rvs(),
                               1000,
                               [0, 0, 0, 0, 4, 5])
            print("Cov x " + str(i) + " accept_rat " + str(accept_rat))
        
    def sample(self, n):
        self.samples = mh(self.stationary,
                          self.proposal_distribution,
                          self.proposal_sampler,
                          n,
                          [0, 0, 0, 0, 4, 5])

    def sample_am(self, n):
        samples, a = am(self.stationary,
                        n,
                        [0, 0, 0, 0, 4, 5],
                        np.eye(6)*2,
                        1)
        self.samples = samples
        return samples, a

    def autocorr(self, x):
        return np.correlate(x, x, mode='full')[len(x)-1:]

    def draw(self, samples):
        xs = self.xs
        ys = self.ys
        
        b00s = [x[0] for x in samples]
        b10s = [x[1] for x in samples]
        b0s = [x[2] for x in samples]
        b1s = [x[3] for x in samples]
        sigmas = [x[4] for x in samples]
        switches = [x[5] for x in samples]

        b00 = np.mean(b00s)
        b10 = np.mean(b10s)
        b0 = np.mean(b0s)
        b1 = np.mean(b1s)
        sigma = np.mean(sigmas)
        switch = np.mean(switches)

        plt.plot(xs, ys, 'ro')
        plt.plot([min(xs), switch], [b0 + min(xs)*b00, b0 + switch*b00])
        plt.plot([switch, max(xs)], [b1 + switch*b10, b1 + max(xs)*b10])
        plt.show()

        # plt.figure(1)
        # plt.subplot(221)
        # plt.plot(range(len(samples)), self.autocorr(b00s))
        # plt.title('autocorelation b00')

        # plt.subplot(222)
        # plt.plot(range(len(samples)), self.autocorr(b10s))
        # plt.title('autocorelation b10')

        # plt.subplot(223)
        # plt.plot(range(len(samples)), self.autocorr(b0s))
        # plt.title('autocorelation b0')

        # plt.subplot(224)
        # plt.plot(range(len(samples)), self.autocorr(b1s))
        # plt.title('autocorelation b1')
        # plt.show()

        plt.figure(1)
        plt.subplot(221)
        plt.plot(range(len(samples)), b00s)
        plt.title('b00')

        plt.subplot(222)
        plt.plot(range(len(samples)), b10s)
        plt.title('b10')

        plt.subplot(223)
        plt.plot(range(len(samples)), b0s)
        plt.title('b0')

        plt.subplot(224)
        plt.plot(range(len(samples)), b1s)
        plt.title('b1')
        plt.show()

        
xs = list(range(30))
ys = np.concatenate([np.linspace(0, 15, 20) + np.random.normal(0, 1, 20),
                     np.random.normal(15, 0.51, 10)])

# xs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# ys = [1, 1, 1, 1, 1, 2, 3, 4, 5]

# xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# ys = np.array([1, 1, 1, 2, 1, 2, 3, 4, 5, 6]) + np.random.normal(0, 1, 10)

blr = Blr_1break(xs, ys)
# samples, a =  blr.sample(5000)
# blr.draw(samples)

samples, a = blr.sample_am(10000)
print(a)
blr.draw(samples)


