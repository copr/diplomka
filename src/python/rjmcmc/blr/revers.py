import sys

import numpy as np

from numpy.random import uniform


class Rjmcmc:
    '''
    trida ktera zastitute cely reversible jump markov chain
    monte carlo algoritmus
    '''

    def __init__(self, move_factory, mcmc_factory, stationary_factory):
        self.move_factory = move_factory
        self.mcmc_factory = mcmc_factory
        self.stationary_factory = stationary_factory
        self.trans_steps = 0
        self.norm_steps = 0

    def step(self, previous_sample):
        k, theta = previous_sample
        u = uniform()
        moves = [m for m in self.move_factory.get_moves_from(k) if m.can_move(k) > 0]
        trans_probability = 0
        for m in moves:
            trans_probability += m.probability_of_this_move(previous_sample)
        if u < trans_probability:
            self.trans_steps += 1
            uu = uniform(0, trans_probability)
            M = None
            prob = 0
            for m in moves:
                if prob < uu < prob + m.probability_of_this_move(previous_sample):
                    M = m
                    break
                prob += m.probability_of_this_move(previous_sample)

            up, down, a, new = self.trans_step(M, previous_sample)
            return new
        else:
            self.norm_steps += 1
            if (k+1)*2 + 3 is not len(theta):
                print(k)
                print(previous_sample)
                raise Exception("wrong length")
            return (k, self.mcmc_factory.get_mcmc(k).step(theta))

    def trans_step(self, move, previous_sample):
        (k, theta) = previous_sample

        new_sample, u, newu, det_jacobian = move.transform(previous_sample)

        (new_k, new_theta) = new_sample
        up = np.prod([
            self.stationary_factory.get_stationary(new_k).pdf(new_theta),
            move.probability_of_this_move(new_sample),
            move.probability_of_help_rvs(new_sample, newu)])
        down = np.prod([
            self.stationary_factory.get_stationary(k).pdf(theta),
            move.probability_of_this_move(previous_sample),
            move.probability_of_help_rvs(previous_sample, u)])

        a = det_jacobian*up/down
        u = uniform()

        if u < a:
            return (up, down, a, new_sample)
        else:
            return (up, down, a, previous_sample)

    def sample(self, n, first_sample):
        samples = [first_sample]
        for i in range(1, n):
            new_sample = self.step(samples[i-1])
            samples.append(new_sample)
            # progress bar
            sys.stdout.write("\r\t%.0f%% Done" % (100*i/n))
            sys.stdout.flush()
        # progress konec
        sys.stdout.write("\r\t100% Done\n")
        sys.stdout.flush()
        return samples
