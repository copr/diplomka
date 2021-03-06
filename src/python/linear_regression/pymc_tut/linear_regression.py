from pymc import deterministic, Uniform, Normal, Model, MCMC
from load_data import load
from matplotlib.pyplot import plot, show
import numpy as np


xs = np.array([2,3,4,5,6,7])
ys = np.array([1.5, 1.8, 1.9, 2.3, 2.5, 2.8])

data = load('data/bran_body_weight.txt', 33)
converted_data = [(float(x[0]), float(x[1]), float(x[2])) for x in data]
xs = [x[1] for x in converted_data]
ys = [x[2] for x in converted_data]
    

b0 = Normal("b0", 0, 0.0003)
b1 = Normal("b1", 0, 0.0003)

err = Uniform("err", 0, 500)

x_weight = Normal("weight", 0, 1, value=xs, observed=True)

@deterministic(plot=False)
def pred(b0=b0, b1=b1, x=x_weight):
    return b0 + b1*x

y = Normal("y", mu=pred, tau=err, value=ys, observed=True)

model = Model([pred, b0, b1, y, err, x_weight])

m = MCMC(model)
m.sample(burn=2000, iter=10000)

bb0 = sum(m.trace('b0')[:])/len(m.trace('b0')[:])
bb1 = sum(m.trace('b1')[:])/len(m.trace('b1')[:])
err = sum(m.trace('err')[:])/len(m.trace('err')[:])

plot(xs, ys)
plot([min(xs), max(xs)], [bb0 + bb1*min(xs), bb0 + bb1*max(xs)])
show()
