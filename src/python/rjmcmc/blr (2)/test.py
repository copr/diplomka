import numpy as np
from blr import Blr

xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
ys = np.array([1, 1, 1, 1, 1, 2, 4, 6, 8, 10, 9, 8,
               7, 6, 5, 8, 10, 12, 14, 16, 18]) + np.random.normal(0, 2, 21)

n_breaks = 1

# prior_mu_2breaks = np.ones(3 + 2*n_breaks)
# prior_mu[1] = 4
# prior_mu[5] = 5
# prior_mu[6] = 10

# 0 b0, 1 sigma, 2 b10, 3 b11, 4 b12, 5 b13, 6 switch1, 7 switch2, 8 switch3
prior_mu = np.zeros(3 + 2*n_breaks)
# prior_mu[1] = 4
# prior_mu[6] = 5
# prior_mu[7] = 10
# prior_mu[8] = 15

# b0, sigma, b10, b11, b12, switch1, switch2

# prior_cov_2breaks = np.array([[5, 0, 0, 0, 0, 0, 0],
#                       [0, 3, 0, 0, 0, 0, 0],
#                       [0, 0, 3, 0, 0, 0, 0],
#                       [0, 0, 0, 3, 0, 0, 0],
#                       [0, 0, 0, 0, 2, 0, 0],
#                       [0, 0, 0, 0, 0, 5, 0],
#                       [0, 0, 0, 0, 0, 0, 5]])
prior_cov = np.eye(3+2*n_breaks)*30
# prior_cov[6, 6] = 3
# prior_cov[7, 7] = 3
# prior_cov[8, 8] = 3
# prior_cov[4, 4] = 10

blr = Blr(xs, ys, n_breaks, prior_mu, prior_cov)
