import sys
import numpy as np
import numpy.random as dist
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as st
from scipy.stats import multivariate_normal as nd_norm

#TODO: jak pocitat stationary
#SUGESTION: asi bude nej, kdyz b, sig, breaks bude v jednom listu treba,
# ale kazdy bude jeste zvlast list nebo tuple

pom1 = 0
pom2 = 0
pom3 = 0

def likelihood(xs, ys, b, sig, breaks):
    ''' 
    likelihood funkce pro linearni regresi s ruznym poctem zlomu
    xs - vektor nezavislych promennych
    ys - vektor zavislych promenych
    sig - rozptyl
    breaks - body zlomu mezi prokladanymy primkami
    '''
    global pom1
    global pom2
    global pom3

    lik = 1
    n = len(xs)
    
    for i in range(n):
        x = xs[i]
        y = ys[i]
        # moc se mi to nelibi, mozna pres list comprehnsiony?
        if x > breaks[len(breaks)-1] :
            pom1 += 1
            # bud je x vetsi nez posledni zlomovy bod
            bracket = y - (x*b[2*len(breaks)] + b[2*len(breaks)+1])
        else:
            pom3 += 1
            # a nebo je mensi nez nejaky break
            for i in range(len(breaks)):
                if x < breaks[i]:
                    pom2 += 1
                    # b1 a b0 jsou za sebou v beckach
                    bracket = y - (x*b[2*i] + b[2*i+1])
                    break
        lik *= sig**(-1/2)*np.exp(-bracket**2/(2*sig))
    return lik

# def likelihood_plot(xs, b, sig, breaks):
#     ys = np.array(len(xs))
#     for i in range(len(xs)):
#         if xs[i] < breaks:
#             bracket = y - (x*b[0] + b[1])
#             ys[i] = sig**(-1/2)*np.exp(-bracket**2/(2*sig))
#         else:
#             bracket = y - (x*b[2] + b[3])
#             ys[i] = sig**(-1/2)*np.exp(-bracket**2/(2*sig))
#     return ys

class rjmcmc:
    def __init__(self):
        print("setup")
        self.xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.ys = [0.5, 0.7, 0.9, 1.1, 1.3, 4.5, 4.7, 4.9, 5.1, 5.3] # + st.norm(0, 0.5).rvs(len(self.xs))

    def prior_b(self, bs):
        return nd_norm(np.ones(len(bs)), np.eye(len(bs))).pdf(bs)

    def prior_sig(self, sig):
        return st.uniform(0,10).pdf(sig)

    def prior_breaks(self, breaks):
        return nd_norm(5, np.eye(len(breaks))*0.5).pdf(breaks)

    def stationary(self, x):
        return likelihood(self.xs, self.ys, x[0], x[1], np.atleast_1d(x[2]))* \
            self.prior_b(x[0])* \
            self.prior_sig(x[1])* \
            self.prior_breaks(np.atleast_1d(x[2]))

    def sample_bs(self, bs):
        return nd_norm(bs, np.eye(len(bs))*0.5).rvs()

    def sample_sigs(self, sig):
        return np.abs(nd_norm(sig, 1).rvs())

    def sample_breaks(self, breaks):
        return nd_norm(breaks, 0.5*np.eye(len(np.atleast_1d(breaks)))).rvs()

    def sampler(self, x):
        return [self.sample_bs(x[0]), self.sample_sigs(x[1]),
                self.sample_breaks(x[2])]

    def sampler_probability_bs(self, bs1, bs2):
        return nd_norm(bs1, np.eye(len(bs1))*0.5).pdf(bs2)

    def sampler_probability_sigs(self, sig1, sig2):
        return nd_norm(sig1, 1).pdf(sig2)

    def sampler_probability_breaks(self, breaks1, breaks2):
        return nd_norm(breaks1, 0.5*np.eye(len(breaks1))).pdf(breaks2)

    def sampler_probability(self, x, y):
        bs1, sig1, breaks1 = x
        bs2, sig2, breaks2 = y
        return self.sampler_probability_bs(bs1, bs2)* \
            self.sampler_probability_sigs(sig1,sig2)* \
            self.sampler_probability_breaks(np.atleast_1d(breaks1),
                                            np.atleast_1d(breaks2))

    def sample(self):
        n = 10000
        init_sample = [(0,0,0,0), 1, 0]
        samples = [init_sample]
        previous_sample = init_sample
        for i in range(n):
            GENERATE_NEW_BREAK = 0
            REMOVE_BREAK = 0
            SAMPLE = 1
            u = st.uniform.rvs()
            if u < GENERATE_NEW_BREAK:
                print('generuju novy break')
            elif u > GENERATE_NEW_BREAK and u < REMOVE_BREAK:
                print('odstranuju break')
            elif u > REMOVE_BREAK and u < SAMPLE:
                u2_bono = st.uniform.rvs()
                new_proposal = self.sampler(previous_sample)
                nominator = self.stationary(new_proposal)*self.sampler_probability(
                    previous_sample, new_proposal)
                denominator = self.stationary(previous_sample)*self.sampler_probability(
                    new_proposal, previous_sample)
                if u2_bono < nominator/denominator:
                    samples.append(new_proposal)
                    previous_sample = new_proposal
                else:
                    samples.append(previous_sample)
            sys.stdout.write("\r\t%.0f%% Done" % (100*i/n))
            sys.stdout.flush()
        # progress konec
        sys.stdout.write("\r\t100% Done\n")
        sys.stdout.flush()
        return samples

rj = rjmcmc()
samples = rj.sample()
b11s = [x[0][0] for x in samples]
b10s = [x[0][1] for x in samples]
b21s = [x[0][2] for x in samples]
b20s = [x[0][3] for x in samples]
sigs = [x[1] for x in samples]
breaks = [x[2] for x in samples]

bb00 = np.mean(b10s)
bb01 = np.mean(b11s)
bb10 = np.mean(b20s)
bb11 = np.mean(b21s)
switch = np.mean(breaks)


plt.plot(rj.xs, rj.ys, 'ro')
plt.plot([min(rj.xs), switch], [bb00 + min(rj.xs)*bb01, bb00 + switch*bb01])
plt.plot([switch, max(rj.xs)], [bb10 + switch*bb11, bb10 + max(rj.xs)*bb11])
plt.show()
