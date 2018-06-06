import theano.tensor as T
import theano
import numpy as np


class TransformationFactory:
    def __init__(self):
        print("init")

    def create_transformation(self, n):
        x = T.vector('x')
        f_list = []
        for i in range(n):
            f_list.append(x[i] + 1)
        f = T.as_tensor_variable(f_list)
        return theano.function([x], f)

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

        assert any([y is not None for y in f_list])
        return theano.function([x], f)

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

        assert any([y is not None for y in f_list])
        return theano.function([x], f)
