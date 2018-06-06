import numpy as np

def sd_pooled(xs, ys):
    # computes pooled standar deviation of xs, ys
    # xs and ys are np.arrays 
    sx = np.std(xs, ddof=1)
    sy = np.std(ys, ddof=1)
    m = xs.size
    n = ys.size
    return np.sqrt(((m-1)*sx**2 + (n-1)*sy**2)/(m+n-2))

def ttest(xs, ys):
    # computes t statistic between xs and ys
    # used to find out if mean values are equal
    # assumptions: ys,xs iid from normal distributino
    #              standart deviations are the same
    xmean = np.mean(xs)
    ymean = np.mean(ys)
    m = xs.size
    n = ys.size
    return (xmean - ymean)/(sd_pooled(xs, ys)*np.sqrt(1/m + 1/n))


xs = np.array([1,4,3,6,5])
ys = np.array([5,4,7,6,10])

if __name__ == "__main__":
    xs = np.array([1,4,3,6,5])
    ys = np.array([5,4,7,6,10])
    print(ttest(xs, ys))
