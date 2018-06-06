import numpy as np

from numpy.random import uniform, randint

from mcmc import Mcmc


class Model:
    '''
    Model by mel byt schopen udelat jeden krok v ramci rjmcmc cyklu

    tedy je potreba
    navrhova distribuce - typu ProposalDistribution
    stacionarni distribuce - s metodou pdf
    
    mozne
    transformace nahoru - typu transformace
    model up - typu model
    transformace dollu - typu transformace
    model down - typu model
    

    metody
    step(previous_sample) - vrati novy vzorek v zavislosti na predchozim
    '''

    def __init__(self,
                 name,
                 proposal_distribution,
                 stationary_distribution,
                 transform_up,
                 model_up,
                 filler_up,
                 model_down,
                 transform_down,
                 filler_down):
        self.name = name
        self.stationary_distribution = stationary_distribution
        self.proposal_distribution = proposal_distribution
        self.transform_up = transform_up
        self.transform_down = transform_down
        self.model_up = model_up
        self.model_down = model_down
        self.filler_up = random_up
        self.filler_down = random_down
        self.mcmc = Mcmc(proposal_distribution, stationary_distribution)
        self.options = ['step']

        if model_up is not None:
            self.options.append('transform_up')
        if model_down is not None:
            self.options.append('transform_down')

    def set_model_up(self, model, transform_up):
        self.options.append('transform_up')
        self.transform_up = transform_up
        self.model_up = model

    def set_model_down(self, model, transform_down):
        self.options.append('transform_down')
        self.transform_down = transform_down
        self.model_down = model
            
    def step(self, previous_sample):
        # tady se uz potrebuju podivat jak se to dela v rjmcmc
        # myslenka je, ze vyberu jednu z moznosti a potom ji
        # zkusim vykonat, pak vratim novy sample a model, ktery se pouzil
        # e.g. kdybych udelal transformaci nahoru tak vracim s novym samplem
        # i "horni model"
        rand_index = randint(len(self.options))
        option = self.options(rand_index)

        if option is 'step':
            new_sample = self.mcmc.step(previous_sample)
            new_model = self
        elif option is 'transform_up':
            new_sample, new_model = self.step_up(previous_sample)
        elif option is 'transform_down':
            new_sample, new_model = self.step_down(previous_sample)
        else:
            raise Exception("Not a viable option")

        return (new_sample, new_model)

    def step_up(self, previous_sample):
        filler_up = self.filler_up.rvs()
        possible_sample, possible_filler, jacobian = self.transform_up(
            previous_sample,
            filler_up)

        numerator = np.prod([
            self.model_up.stationary_distribution.pdf(possible_sample),
            self.model_up.filler_down.pdf(possible_filler),
            1])  # jm(x')

        denominator = np.prod([
            self.stationary_distribution.pdf(previous_sample),
            self.filler_up.pdf(filler_up),
            1])

        u = uniform()
        probability = jacobian*numerator/denominator

        if u < probability:
            return (previous_sample, self.model)
        return (possible_sample, self.model_up)

    def step_down(self, previous_sample):
        filler_down = self.filler_down.rvs()
        possible_sample, possible_filler, jacobian = self.transform_down(
            previous_sample,
            filler_down)

        numerator = np.prod([
            self.model_down.stationary_distribution.pdf(possible_sample),
            self.model_down.filler_up.pdf(possible_filler),
            1])

        denominator = np.prod([
            self.stationary_distribution.pdf(previous_sample),
            self.filler_down.pdf(filler_down)
            1])

        u = uniform()
        probability = jacobian*numerator/denominator

        if u < probability:
            return (previous_sample, self.model)
        return (possible_sample, self.model_down)
