import utils

import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, xs, ys, line_x, line_y, samples):
        self.xs = xs
        self.ys = ys
        self.line_x = line_x
        self.line_y = line_y
        self.samples = samples
        self.ks = []
        self.sorted_samples = {}
        self._sort_samples(samples)

    def plot_hist_jumps(self):
        plt.hist(self.ks)
        plt.show()

    def plot_time_line_jumps(self):
        plt.plot(self.ks)
        plt.show()

    def plot_lines(self, k, every_nth=50, opacity=0.1):
        n = 2*k + 5
        utils.plot_lines(self.xs,
                         self.ys,
                         self.sorted_samples[k],
                         n,
                         every_nth,
                         opacity,
                         self.line_x,
                         self.line_y)

    def plot_hists(self, k, every_nth=50, opacity=0.1):
        n = 2*k + 5
        utils.plot_hists(self.xs,
                         self.ys,
                         self.sorted_samples[k],
                         n,
                         every_nth,
                         opacity,
                         self.line_x,
                         self.line_y)

    def _sort_samples(self, samples):
        self.sorted_samples = {}
        for sample in samples:
            k, theta = sample
            self.ks.append(k)
            if k in self.sorted_samples.keys():
                self.sorted_samples[k].append(theta)
            else:
                self.sorted_samples[k] = [theta]
