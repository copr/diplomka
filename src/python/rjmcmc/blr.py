import sys
import numpy as np
import numpy.random as dist
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as st
from scipy.stats import multivariate_normal as nd_norm

xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ys = [1, 1, 1, 1, 1, 2, 3, 4, 5, 6]  + st.norm(0, 0.5).rvs(len(xs))
n = len(xs)

# def likelihood(xs, ys, b11, b10, b21, b20, sig, s):
#     lik = 1
#     n = len(xs)
#     for i in range(n):
#         x = xs[i]
#         y = ys[i]
#         if x < s:
#             bracket = y - (x*b11 + b10)
#         else:
#             bracket = y - (x*b21 + b20)
#         lik *= sig**(-1/2)*np.exp(-bracket**2/(2*sig))
#     return lik

def likelihood(xs, ys, b, sig, breaks):
    lik = 1
    n = len(xs)
    for i in range(n):
        x = xs[i]
        y = ys[i]
        # moc se mi to nelibi, mozna pres list comprehnsiony?
        if x > breaks[len(breaks)-1]:
            # bud je x vetsi nez posledni zlomovy bod
            bracket = y - (x*b[2*len(breaks)] + b[2*len(breaks)+1])
        else:
            # a nebo je mensi nez nejaky break
            for i in range(len(breaks)):
                if x < breaks[i]:
                    # b1 a b0 jsou za sebou v beckach
                    bracket = y - (x*b[2*i] + b[2*i+1])
                    break
        lik *= sig**(-1/2)*np.exp(-bracket**2/(2*sig))
    return lik

priorb = lambda b: nd_norm([0, 1, 1, 1], np.array([[1,0,0,0], [0, 1, 0, 0],
                                                   [0, 0, 3, 0], [0, 0, 0, 3]])).pdf(b)
priorbreak = lambda s: st.norm(5, 2).pdf(s)
priorsig = lambda s: st.uniform(0,3).pdf(s)
            
post = lambda b, sig, s: likelihood(xs, ys, b, sig, [s])*priorb(b)*priorsig(sig)*priorbreak(s)

cov = np.array([[0.5,0,0,0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]])

proposalb_sampler = lambda b: nd_norm([b[0], b[1], b[2], b[3]], cov).rvs()
proposalb = lambda b1, b2: nd_norm([b1[0], b1[1], b1[2], b1[3]], cov).pdf([b2[0], b2[1], b2[2], b2[3]])
proposalbreak_sampler = lambda s: nd_norm(s, 1).rvs()
proposalbreak = lambda s1, s2: nd_norm(s1, 1).pdf(s2)
proposalsig_sampler = lambda s: np.abs(nd_norm(s, 1).rvs())
proposalsig = lambda s1,s2: nd_norm(np.abs(s1), 1).pdf(s2)

proposal_sampler = lambda b, sig, s: proposalb_sampler(b).tolist() + [proposalsig_sampler(sig), proposalbreak_sampler(s)]
proposal = lambda b1, sig1, s1, b2, sig2, s2: proposalb(b1, b2)*proposalbreak(s1, s2)*proposalsig(s1, s2)

k = 10000
samples = np.zeros(k).tolist()
samples[0] = [1, 1, 1, 1, 1, 5]
for i in range(1, k):
    u = dist.uniform()
    xprev = samples[i-1]
    bprev = xprev[0:4]
    sigprev = xprev[4]
    sprev = xprev[5]
    xprop = proposal_sampler(bprev, sigprev, sprev)
    bprop = xprop[0:4]
    sigprop = xprop[4]
    sprop = xprop[5]
    up = post(bprop, sigprop, sprop)*proposal(bprev, sigprev, sprev, bprop, sigprop, sprop)
    down = post(bprev, sigprev, sprev)*proposal(bprop, sigprop, sprop, bprev, sigprev, sprev)
    if u < up/down:
        samples[i] = xprop
    else:
        samples[i] = xprev
    sys.stdout.write("\r\t%.0f%% Done" % (100*i/k))
    sys.stdout.flush()
# progress konec
sys.stdout.write("\r\t100% Done\n")
sys.stdout.flush()

b11s = [x[0] for x in samples]
b10s = [x[1] for x in samples]
b21s = [x[2] for x in samples]
b20s = [x[3] for x in samples]
sigs = [x[4] for x in samples]
breaks = [x[5] for x in samples]

bb00 = np.mean(b10s)
bb01 = np.mean(b11s)
bb10 = np.mean(b20s)
bb11 = np.mean(b21s)
switch = np.mean(breaks)

plt.plot(xs, ys, 'ro')
plt.plot([min(xs), switch], [bb00 + min(xs)*bb01, bb00 + switch*bb01])
plt.plot([switch, max(xs)], [bb10 + switch*bb11, bb10 + max(xs)*bb11])
plt.show()
