import sys
import copy
import numpy as np


class Mcmc:
    def __init__(self, proposalDistribution, stationaryDistribution):
        self.proposal = proposalDistribution
        self.stationary = stationaryDistribution

    def step(self, previous_sample):
        '''
        Vnitrek mcmc algoritmu, prakticky to co se deje v jedne 
        iteraci tady te verze hastingse
        '''
        proposal_sample = self.proposal.rvs(previous_sample)
        assert proposal_sample is not None
        local_previous_sample = copy.copy(previous_sample)
        final_sample = copy.copy(previous_sample)

        for j, x in enumerate(proposal_sample):
            local_previous_sample[j] = x
            down = np.prod([
                self.stationary.pdf(previous_sample),
                self.proposal.pdf(local_previous_sample, previous_sample)])
            up = np.prod([
                self.stationary.pdf(local_previous_sample),
                self.proposal.pdf(previous_sample, local_previous_sample)])

            u = np.random.uniform()

            if u < up/down:
                final_sample[j] = x
            else:
                local_previous_sample[j] = previous_sample[j]

        return final_sample

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

