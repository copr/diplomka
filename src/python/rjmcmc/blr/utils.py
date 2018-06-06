import matplotlib.pyplot as plt
import numpy as np


def plot(xs, ys, samples, dimension):
    '''
    vykresli data a ziskane primky
    '''
    mean = np.mean(samples, 0)
    ss = [mean[i] for i in range(1, dimension, 2)]
    hs = [mean[i] for i in range(2, dimension, 2)]
    plt.plot(xs, ys, '*')
    plt.plot(ss, hs, 'r')
    plt.show()


def individual_samples(samples, dimension):
    sigmas = [x[0] for x in samples]
    ss = []
    hs = []
    for i in range(1, dimension):
        if i % 2 == 0:
            hs.append([x[i] for x in samples])
        else:
            ss.append([x[i] for x in samples])
    return (ss, hs, sigmas)


def plot_lines(xs, ys, samples, dimension, nth, alpha, origy):
    for i in range(0, len(samples), nth):
        sample = samples[i]
        xss = [sample[i] for i in range(1, dimension, 2)]
        yss = [sample[i] for i in range(2, dimension, 2)]
        plt.plot(xss, yss, 'g', alpha=alpha)
    plt.plot(xs, ys, 'bo')
    plt.plot(xs, origy, 'b')
    xmax = max(xs)
    xmin = min(xs)
    ymax = max(ys)
    ymin = min(ys)
    plt.axis([xmin-1, xmax+1, ymin-1, ymax+1])
    plt.show()

    
def color(i):
    if i == 1:
        return 'go'
    elif i == 3:
        return 'ro'
    elif i == 5:
        return 'ko'
    elif i == 7:
        return 'co'
    elif i == 9:
        return 'yo'
    else:
        raise Exception("error")


def plot_hists(xs, ys, samples, dimension, nth, alpha, origy):
    for i in range(0, len(samples), nth):
        sample = samples[i]
        for j in range(1, dimension, 2):
            x = sample[j]
            y = sample[j+1]
            plt.plot(x, y, color(j), alpha=alpha)
    plt.plot(xs, ys, 'bo')
    plt.plot(xs, origy, 'b')
    xmax = max(xs)
    xmin = min(xs)
    ymax = max(ys)
    ymin = min(ys)
    plt.axis([xmin-1, xmax+1, ymin-1, ymax+1])
    plt.show()


