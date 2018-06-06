from pymc import deterministic, DiscreteUniform, Uniform, Normal, Model, MCMC, graph
from load_data import load
from matplotlib.pyplot import plot, show, hist
from scipy.stats import mode
import numpy as np
import math

# data = load('data/bran_body_weight.txt', 33)
# converted_data = [(float(x[0]), float(x[1]), float(x[2])) for x in data]
# xs = np.array([x[1] for x in converted_data])
# ys = np.array([x[2] for x in converted_data])

xs = np.array([1,2,3,4,5,6,7,8,9,10])
ys = np.array([1,1,1,1,1,2,3,4,5,6]) + np.random.normal(0, 2, len(xs))


b00 = Uniform("b00", -50, 50)
b01 = Uniform("b01", -50, 50)

b10 = Uniform("b10", -50, 50)
b11 = Uniform("b11", -50, 50)

switchpoint = DiscreteUniform("switch", min(xs), max(xs))

err = Uniform("err", 50, 500)

x_weight = Normal("weight", 0, 1, value=xs, observed=True)

@deterministic(plot=False)
def pred(b00=b00, b01=b01, b10=b10, b11=b11, s=switchpoint):
    out = np.empty(len(xs))
    breakk = s
    out[:breakk] = b00 + b01*xs[:breakk]
    out[breakk:] = b10 + b11*xs[breakk:]
    return out

y = Normal("y", mu=pred, tau=err, value=ys, observed=True)

model = Model([pred, b00, b01, b10, b11, switchpoint, y, err])

# g = graph.graph(model)
# g.write_png('graph.png')

m = MCMC(model)
m.sample(burn=1000, iter=10000)

bb00 = np.mean(m.trace('b00')[:])
bb01 = np.mean(m.trace('b01')[:])
bb10 = np.mean(m.trace('b10')[:])
bb11 = np.mean(m.trace('b11')[:])
switch = mode(m.trace('switch')[:])[0][0]

plot(xs, ys, 'ro')
plot([min(xs), switch], [bb00 + min(xs)*bb01, bb00 + switch*bb01])
plot([switch, max(xs)], [bb10 + switch*bb11, bb10 + max(xs)*bb11])
show()
