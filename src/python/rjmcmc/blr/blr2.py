import numpy as np
from scipy.stats import multivariate_normal as normal
from scipy.stats import uniform
from functools import partial

class Blr2:
    '''
    Trida, ktera vytvori stacionarni distribuci pro regresi s n zlomy.
    2 protoze pouzivam jinou parametrizaci.
    sigma = x[0] - rozptyl no :D
    s0 = x[1] - xova souradnice prvniho zlomu
    h0 = x[2] - yova souradnice prvnhio zlomu
    ...
    sn = x[n-2] - xova souradnice nteho zlomu
    hn = x[n-1] - yova souradnice nhteo zlomu
    '''

    def __init__(self, xs, ys, n_breaks):
        '''
        @param xs - xove souradnice dat
        @param ys - yove souradnice dat
        @param n_breaks - pocet zlomu
        '''
        if len(xs) is not len(ys):
            raise RuntimeError("Not matchin dimension")
        self.xs = xs
        self.ys = ys
        self.max_x = max(xs)
        self.min_x = min(xs)
        self.n = 2*n_breaks + 5
        self.n_samples = len(xs)
        self.h_prior = normal(np.zeros(int((self.n-1)/2)),
                              100*np.eye(int((self.n-1)/2)))
        self.sigma_prior = normal(0, 3)
        self.n_breaks = n_breaks
        
    def prior_s(self, theta):
        '''
        Apriorni rozdeleni na thetaovych souradnicich. Tedy melo by platit
        ss < s1 < s2 < ... < sn < sf. Je to tak nastaveno z toho duvodu,
        aby bylo dodrzeni poradi
        '''
        # taky si nejsem jisty jestli prochazim vsechny
        x_coordinates = [theta[i] for i in range(1, self.n, 2)]
        previous = x_coordinates[0]
        for i in range(1, len(x_coordinates)):
            if previous > x_coordinates[i]:
                return 0
            previous = x_coordinates[i]

        if x_coordinates[0] < self.min_x - 0.1:
            return 0
        if x_coordinates[len(x_coordinates) - 1] > self.max_x + 0.1:
            return 0
        return 1

    def prior_h(self, theta):
        '''
        Apriorni rozdeleni na yovych souradnicich. Je teda co nejvic
        neinformativni, tedy pro vsechny h plati, ze h ~ N(0, 100)
        '''
        # tady se trochu bojim ze neprojdu vsechny
        # jestli se dostanu za hranici tak se to rychle odhali :D
        y_coordinates = [theta[i] for i in range(2, self.n, 2)]
        return self.h_prior.pdf(y_coordinates)

    def prior_sigma(self, theta):
        '''
        Apriorni rozdeleni na rozptylu. Zas jen nake neinformativni a s
        nulovou pravdepodobnosti na sigmach mensi nez 0
        '''
        if theta[0] > 0:
            return self.sigma_prior.pdf(theta[0])
        return 0

    def likelihood(self, theta):
        '''
        Spocita likelihood hustotu pro dany vzorek
        '''
        assert len(theta) == self.n

        suma = 0
        for i, xi in enumerate(self.xs):
            yi = self.ys[i]
            for j in range(1, self.n-2, 2):
                break1 = (theta[j], theta[j+1])
                break2 = (theta[j+2], theta[j+3])

                if break1[0] <= xi < break2[0]:
                    suma += self.prob_sum(xi, yi, break1, break2)

            # tohle je pro pripad ze xi je pred nebo za body urcujicimi primku
            # stava se to :D
            if xi < theta[1]:
                break1 = (theta[1], theta[2])
                break2 = (theta[3], theta[4])
                suma += self.prob_sum(xi, yi, break1, break2)

            if xi > theta[self.n - 2]:
                break1 = (theta[self.n-4], theta[self.n-3])
                break2 = (theta[self.n-2], theta[self.n-1])
                suma += self.prob_sum(xi, yi, break1, break2)

        try:
            exp = np.exp(-suma/(2*theta[0]))
            bs = theta[0]**(-len(self.xs)/2)
            return bs * exp
        except FloatingPointError:
            print()
            print('theta0 ' + str(theta[0]))
            print('suma ' + str(suma))
            return 0

    def prob_sum(self, x, y, break1, break2):
        '''
        pomocna funkce, co mi spocita jeden vyraz v exponenciale, jakoze v
        Normalnim rozdeleni hore
        '''

        # a = (break2[1] - break1[1])/(break2[0]-break1[0])
        # b = break2[1] - break2[0]*a
        x1, y1 = break1
        x2, y2 = break2
        est = (x - x1)*(y2 - y1)/(x2 - x1) + y1
        return (y - est)**2

    def pdf(self, theta):
        if len(theta) is not self.n:
            print(theta)
            raise Exception("Co to kurva")

        prior_probs = np.prod([self.prior_h(theta),
                               self.prior_s(theta),
                               self.prior_sigma(theta)])

        # netkere vzorky budou mit nulovou pravdepodobnost
        # uz kvuli apriornimu rozdeleni, proto to checknu
        # at se nemusi pocitat likelihood ten v zavislosti
        # na datech muze byt dost narocny spocitat
        if prior_probs == 0:
            return 0
        return np.prod([prior_probs,
                        self.likelihood(theta)])

    def generate_first_sample(self):
        '''
        vygeneruje nejaky vzorek, ktery nema pravdepodobnost nula
        '''
        minimum = min(self.xs)
        maximum = max(self.xs)
        first_sample = np.zeros(self.n)

        first_sample[0] = 1

        for i, x in enumerate(np.linspace(minimum, maximum, (self.n-1)/2)):
            first_sample[2*i + 1] = x
            first_sample[2*i + 2] = np.random.normal(0, 3)

        if not self.pdf(first_sample) > 0:
            print("First sample: " + str(first_sample) +
                  " has zero probability")
            print("prior h " + str(self.prior_h(first_sample)))
            print("prior s " + str(self.prior_s(first_sample)))
            print("prior sigma " + str(self.prior_sigma(first_sample)))
            print("likelihood " + str(self.likelihood(first_sample)))
            return self.generate_first_sample()

        return first_sample
