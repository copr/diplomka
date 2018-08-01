class Manager:
    def __init__(self, stationaryDistributionFactory, moveFactory, mcmcFactory):
        self.stats = stationaryDistributionFactory
        self.moves = moveFactory
        self.mcmcms = mcmcFactory

    
