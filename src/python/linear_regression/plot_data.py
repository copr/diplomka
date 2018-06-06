from matplotlib.pyplot import plot, show

def plot_lg(bs, intervals):
    i = 0
    for b in bs:
        plot([intervals[i][0], intervals[i][1]], [intervals[i][0] + b[0], b[0] + b[1]*intervals[i][1]])
        i += 1
    show()
