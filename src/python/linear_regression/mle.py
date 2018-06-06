from load_data import load
from matplotlib import pyplot as plt

def mle(xs, ys):
    n = len(xs)
    a = 0
    for i in range(n):
        a += ys[i]/xs[i]
    return a

def mle2(xs, ys):
    n = len(xs)
    a = 0
    b = 0
    xmean = sum(xs)/n
    ymean = sum(ys)/n
    up1 = 0
    up2 = 0
    down = 0
    for i in range(n):
        # up += (ymean - ys[i])*(xmean - xs[i])
        # down += (xmean - xs[i])**2
        up1 += ys[i]*xs[i]
        up2 += ymean*xs[i]
        down += xs[i]*(xmean - xs[i])
    a = (up2-up1)/down
    for i in range(n):
        b += ys[i] - a*xs[i]
    b = b/n
    return (a, b)

def mse(ys, predicted):
    suma = 0
    for i in range(len(ys)):
        error = (ys[i] - predicted[i])**2
        suma += error
    return suma/len(ys)


if __name__ == "__main__":
    data = load('data/bran_body_weight.txt', 33)
    converted_data = [(float(x[0]), float(x[1]), float(x[2])) for x in data]
    xs = [x[1] for x in converted_data]
    ys = [x[2] for x in converted_data]

    # xs = [1,2,3,4,5,6,7]
    # ys = [1.5, 1.8, 1.9, 2.3, 2.5, 2.8, 2.8]

    (a,b) = mle2(xs, ys)
    print(a)
    print(b)
    plt.plot(xs, ys, 'ro')
    plt.plot([0, max(xs)], [b, a*max(xs) + b])
    plt.show()
        
    
