import numpy as np
import tensorly as tl
import random as rd

from tensorly.contrib.decomposition import tensor_train_cross
from tensorly.decomposition import tensor_train

N = 5 * 5 * 5
sparsity = 0.8
entry = np.zeros(N)
for i in range(N):
    r = rd.random()
    if r > sparsity:
        entry[i] = rd.random()

tensor = tl.tensor(entry.reshape(5,5,5))
rank = [1, 3, 3, 1]
factors = tensor_train(tensor, rank)
rec_tensor = tl.tt_to_tensor(factors)
error = tl.norm(rec_tensor - tensor)/tl.norm(tensor)
print(error)

