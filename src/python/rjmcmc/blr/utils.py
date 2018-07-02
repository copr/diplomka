import matplotlib.pyplot as plt
import numpy as np


def generate_samples(line_xs, line_ys, n, var=0.5):
    max_n = 10000
    if n > max_n:
        raise Exception("Can't provide so many")
    if len(line_xs) is not len(line_ys):
        raise Exception("Vectors must be of the same length")
    xs = np.linspace(line_xs[0], line_xs[-1], max_n)
    ys = np.random.normal(0, var, max_n)
    step = int(max_n/n)
    for i in range(0, max_n, step):
        x = xs[i]
        for j in range(0, len(line_xs)-1):
            x1 = line_xs[j]
            x2 = line_xs[j+1]
            y1 = line_ys[j]
            y2 = line_ys[j+1]
            a = (y2 - y1)/(x2 - x1)
            b = y1 - a*x1
            if line_xs[j] <= x <= line_xs[j+1]:
                ys[i] += a*x + b

    xxx = [xs[i] for i in range(0, max_n, step)]
    yyy = [ys[i] for i in range(0, max_n, step)]
    assert(len(xxx) == n)
    assert(len(yyy) == n)

    return (xxx, yyy)


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


def plot_lines(xs, ys, samples, dimension, nth, alpha, origx, origy):
    for i in range(0, len(samples), nth):
        sample = samples[i]
        xss = [sample[i] for i in range(1, dimension, 2)]
        yss = [sample[i] for i in range(2, dimension, 2)]
        plt.plot(xss, yss, 'g', alpha=alpha)
    plt.plot(xs, ys, 'bo')
    plt.plot(origx, origy, 'b')
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


def plot_hists(xs, ys, samples, dimension, nth, alpha, origx, origy):
    for i in range(0, len(samples), nth):
        sample = samples[i]
        for j in range(1, dimension, 2):
            x = sample[j]
            y = sample[j+1]
            plt.plot(x, y, color(j), alpha=alpha)
    plt.plot(xs, ys, 'bo')
    plt.plot(origx, origy, 'b')
    xmax = max(xs)
    xmin = min(xs)
    ymax = max(ys)
    ymin = min(ys)
    plt.axis([xmin-1, xmax+1, ymin-1, ymax+1])
    plt.show()


