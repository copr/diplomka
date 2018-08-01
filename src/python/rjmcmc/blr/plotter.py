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
        n = len(self.sorted_samples.keys())
        m = len(self.samples)
        probs = [len(self.sorted_samples[key])/m for key in self.sorted_samples]
        plt.bar(list(self.sorted_samples.keys()), probs, align='center')
        plt.xticks(list(self.sorted_samples.keys()))
        plt.xlabel("m")
        plt.ylabel("aproximace aposteriorní pravdepodobnosti modelu")
        plt.show()

    def plot_time_line_jumps(self):
        plt.plot(self.ks)
        plt.xlabel("i-ta iterace")
        plt.ylabel("m")
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
