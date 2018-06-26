import numpy as np

from scipy.stats import multivariate_normal as normal

from mcmc import Mcmc
from ProposalDistribution2 import ProposalDistribution2


class McmcFactory:
    def __init__(self, stationaryDistributionFactory):
        self.dimension2mcmc = {}
        self.stat_factory = stationaryDistributionFactory

    def get_mcmc(self, k):
        if k in self.dimension2mcmc.keys():
            return self.dimension2mcmc[k]
        blr = self.stat_factory.get_stationary(k)
        prop = ProposalDistribution2(normal(np.zeros(blr.n), 5*np.eye(blr.n)))
        mcmc = Mcmc(prop, blr)
        self.dimension2mcmc[k] = mcmc
        return mcmc
