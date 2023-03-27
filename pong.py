# http://karpathy.github.io/2016/05/31/rl/

import numpy as np

h = np.dot(W1, x) # compute hidden layer neuron activation
h[h<0] = 0 # ReLu nonlinearity: threshold at zero
logp = np.dot(W2, h) # compute log probability of going up
p = 1.0 / (1.0 + np.exp(-logp)) # sigmoid function (gives probability of going up)

