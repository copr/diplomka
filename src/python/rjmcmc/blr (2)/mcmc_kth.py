import sys

import numpy as np

from mcmc import Mcmc


class Mcmc_kth:
    def __init__(self,
                 proposalDistribution1,
                 proposalDistribution2,
                 stationaryDistribution,
                 k):
        self.mcmc1 = Mcmc(proposalDistribution1, stationaryDistribution)
        self.mcmc2 = Mcmc(proposalDistribution2, stationaryDistribution)
        self.k = k
        self.k_counter = 0

    def step(self, previous_sample):
        self.k += 1
        if self.k_counter % self.k is not 0:
            return self.mcmc1.step(previous_sample)
        else:
            return self.mcmc2.step(previous_sample)

    def sample(self, n, first_sample):
        dimension = len(first_sample)
        samples = np.empty((n, dimension))
        samples[0] = first_sample
        # mozna generator?
        for i in range(1, n):
            samples[i] = self.step(samples[i-1])
            # progress bar
            sys.stdout.write("\r\t%.0f%% Done" % (100*i/n))
            sys.stdout.flush()
        # progress konec
        sys.stdout.write("\r\t100% Done\n")
        sys.stdout.flush()
        return samples
    
