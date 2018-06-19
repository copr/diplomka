import theano.tensor as T
import theano
import numpy as np

from scipy.stats import multivariate_normal as normal
from scipy.stats import uniform

from abstractMoveFactory import AbstractMoveFactory
from move import Move
from proposalDistribution import ProposalDistribution


class BlrMoveFactory(AbstractMoveFactory):
    def __init__(self):
        self.dimension_to_moves = {}

    def get_moves_from(self, k):
        if k in self.dimension_to_moves.keys():
            return self.dimension_to_moves[k]

        moves = []
        u = uniform(0, 1)
        n = normal(0, 1)
        # moves up
        for i in range(k+1):
            t_up, j_up = self._create_transformation_up(k, i)
            t_down, j_down = self._create_transformation_down(k+1, i)
            u_gen_up = ProposalDistribution(lambda x: u.pdf(x[0])*n.pdf(x[1]),
                                            lambda: [u.rvs(), n.rvs()])
            move = Move(k, k+1, 0.05/(k+1), 0.05/(k+1),
                        t_up, t_down, j_up, j_down,
                        u_gen_up, None, 2, 0)
            moves.append(move)
        # moves down
        for i in range(k):
            t_up, j_up = self._create_transformation_up(k-1, i)
            t_down, j_down = self._create_transformation_down(k, i)
            u_gen_up = ProposalDistribution(lambda x: u.pdf(x[0])*n.pdf(x[1]),
                                            lambda: [u.rvs(), n.rvs()])
            move = Move(k-1, k, 0.05/k, 0.05/k,
                        t_up, t_down, j_up, j_down,
                        u_gen_up, None, 2, 0)
            moves.append(move)

        self.dimension_to_moves[k] = moves
        return moves

    def _create_transformation_up(self, k, m):
        '''
        k - pocet zlomu
        m - ktery zlom zlomit :D,
            e.g 0 - novy zlom mezi 0. a 1.
                1 - novy zlom mezi 1. a 2.
        '''
        if m < 0 or m > k:
            raise Exception("Nemozno zlomit takto")

        x = T.vector('x')
        # k*2 - kazdy zlom ma dva body, 1 - sigma, 4 - koncove body,
        # 2 - nahodne veliciny
        n = k*2 + 1 + 4 + 2
        f_list = [None for _ in range(n)]
        f_list[0] = x[0]
        break1 = 1 + 2*m + 2
        for i in range(1, break1):
            f_list[i] = x[i]

        f_list[break1] = x[break1-2] + (x[break1]-x[break1-2])*x[n-2]
        f_list[break1+1] = x[break1-1] + (x[break1+1]-x[break1-1])*x[n-2] + x[n-1]

        for i in range(break1+2, n):
            f_list[i] = x[i-2]

        f = T.as_tensor_variable(f_list)
        jacobian = theano.function([x], theano.gradient.jacobian(f, x))

        assert any([y is not None for y in f_list])
        return theano.function([x], f), jacobian

    def _create_transformation_down(self, k, m):
        '''
        k - pocet zlomu
        m - ktery zlom vymazat :D,
            e.g 0 - vymazat zlom mezi 0. a 2.
                1 - vymazat zlom mezi 1. a 3.
        '''
        if m < 0 or m >= k:
            raise Exception("Nemozno vymazat takto")

        x = T.vector('x')
        # k*2 - kazdy zlom ma dva body, 1 - sigma, 4 - koncove body,
        n = k*2 + 1 + 4
        f_list = [None for _ in range(n)]
        f_list[0] = x[0]
        break1 = 1 + 2*m + 2

        for i in range(1, break1):
            f_list[i] = x[i]

        for i in range(break1, n-2):
            f_list[i] = x[i+2]

        f_list[n-2] = (x[break1] - x[break1-2])/(x[break1+2] - x[break1-2])
        f_list[n-1] = x[break1+1] - x[break1-1] - (x[break1+3] - x[break1-1])*(x[break1] - x[break1-2])/(x[break1+2] - x[break1-2])

        f = T.as_tensor_variable(f_list)
        jacobian = theano.function([x], theano.gradient.jacobian(f, x))

        assert any([y is not None for y in f_list])
        return theano.function([x], f), jacobian


f = BlrMoveFactory()
mvs = f.get_moves_from(1)
sample = (1, [1, 1, 1, 5, 5, 7, 7])
(newx, u, newu, det_jacobian) = mvs[0].transform(sample)
