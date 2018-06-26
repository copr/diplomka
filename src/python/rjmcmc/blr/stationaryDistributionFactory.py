from blr2 import Blr2 as Blr


class StationaryDistributionFactory:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.dimension2stationary = {}

    def get_stationary(self, k):
        if k in self.dimension2stationary.keys():
            return self.dimension2stationary[k]
        blr = Blr(self.xs, self.ys, k)
        self.dimension2stationary[k] = blr
        return blr
