import numpy as np
import matplotlib.pyplot as plt

p = np.array([i*0.1 + 0.05 for i in range(10)])
prior = np.array([1, 5.2, 8, 7.2, 4.6, 2.1, 0.7, 0.1, 0, 0])
prior = prior/np.sum(prior)
