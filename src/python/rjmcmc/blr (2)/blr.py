import numpy as np
from scipy.stats import multivariate_normal as normal
from functools import partial


class Blr:
    '''
    Trida, ktera mi vytvori stacionarni distribuci pro regresi s n zlomy.
    b0 = x[0]
    sigma = x[1], odchylka
    b1s = x[2:3+n_breaks],
    switches = x[3+n_breaks:], zlomy
    '''
    def __init__(self, xs, ys, n_breaks, prior_mu, prior_cov):
        self.xs = xs
        self.ys = ys
        self.n_breaks = n_breaks
        self.prior_mu = prior_mu
        self.prior_cov = prior_cov
        self.dimension = 3+2*n_breaks

        self.pdf = self.blr_nbreaks(xs, ys)

    def blr_nbreaks(self, xs, ys):
        '''
        vraci funkci co spocita posterior hustotu pro bayes linear regresi
        s n zlomy
        ys, xs - ys pozorovane hodnoty, xs nezavisle hodnoty
        b0 - y=b00*x+b0
        sigma - rozpytyl
        b1s - n+1 sklonu primky
        switches - n zlomu primky
        '''
        # prvni budde vzdycky b0 a sigma pak b1s a swithces
        # takze delka jednoho samplu by mela byt vzdycky
        # 1 + 1 + (n+1) + n = 3+2n
        prior = partial(normal(self.prior_mu, self.prior_cov).pdf)
        n_breaks = self.n_breaks

        def prob_density(x):
            assert len(x) == self.dimension
            
            b0 = x[0]
            sigma = x[1]
            b1s = x[2:3+n_breaks]
            switches = x[3+n_breaks:]
            # pridam si nekonecno do switchu at muzu tesstovat
            # jen z jedne strany
            switches = np.append(switches, np.inf)
            prob = 0
            n = len(xs)
            for i, xi in enumerate(xs):
                # v jake hodnote zacinam
                for j, switch in enumerate(switches):
                    if j == 0:
                        a = b0
                    else:
                        a = a + b1s[j-1]*switches[j-1] - b1s[j]*switches[j-1]
                        if xi < switch:
                            prob += (ys[i] - (xi*b1s[j]+a))**2
                            break
                sigma = abs(sigma)
                if sigma < 0:
                    raise Exception("sigma < 0")
                prob = (sigma)**(-n/2) * np.exp(-prob/(2*sigma))
                return prob*prior(x)
            return prob_density
