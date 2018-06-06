import theano.tensor as T
import theano
import numpy as np

from transformationFactory import TransformationFactory

tf = TransformationFactory()

x = [1, 1, 1, 3, 3, 5, 5, 0.5, 0.21]

f = tf._create_transformation_up(1, 1)
g = tf._create_transformation_down(2, 1)

# x = T.vector('x')

# h0 = x[0]
# h1 = x[1]
# h2 = x[2]
# h3 = x[1] + (x[3]-x[1])*x[5]
# h4 = x[2] + (x[4]-x[2])*x[5] + x[6]
# h5 = x[3]
# h6 = x[4]

# h = T.as_tensor_variable([h0, h1, h2, h3, h4, h5, h6])

# hh = theano.function([x], h)

# hj = theano.gradient.jacobian(h, x)

# h_jacobian = theano.function([x], hj)

# y = T.vector('y')
# g0 = y[0]
# g1 = y[1]
# g2 = y[2]
# g3 = y[5]
# g4 = y[6]
# g5 = (y[3] - y[1])/(y[5] - y[1])
# g6 = y[4] - y[2] - (y[6] - y[2])*(y[3] - y[1])/(y[5] - y[1])


# g = T.as_tensor_variable([g0, g1, g2, g3, g4, g5, g6])

# gg = theano.function([y], g)

# gj = theano.gradient.jacobian(g, y)

# g_jacobian = theano.function([y], gj)


# i = T.vector('i')
# e0 = i[0]
# e1 = i[1]
# e2 = i[2]
# e3 = i[3]
# e4 = i[4]
# e5 = i[3] + (i[5] - i[3])*i[7]
# e6 = i[4] + (i[6] - i[4])*i[7] + i[8]
# e7 = i[5]
# e8 = i[6]

# e = T.as_tensor_variable([e0, e1, e2, e3, e4, e5, e6, e7, e8])

# ee = theano.function([i], e)

# ej = theano.gradient.jacobian(e, i)

# e_jacobian = theano.function([i], ej)

# j = T.vector('j')

# d0 = j[0]
# d1 = j[1]
# d2 = j[2]
# d3 = j[3]
# d4 = j[4]
# d5 = j[7]
# d6 = j[8]
# d7 = (j[5] - j[3])/(j[7] - j[3])
# d8 = j[6] - j[4] - (j[8] - j[4])*(j[5] - j[3])/(j[7] - j[3])

# d = T.as_tensor_variable([d0, d1, d2, d3, d4, d5, d6, d7, d8])

# dd = theano.function([j], d)

# dj = theano.gradient.jacobian(d, j)

# d_jacobian = theano.function([j], dj)
