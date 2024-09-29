import numpy as np
import scipy as sp
from scipy.sparse import random

from scipy.linalg.interpolative import interp_decomp
from scipy.linalg.interpolative import reconstruct_interp_matrix

S = random(10, 10, 0.9)
dS = S.toarray()

k, idx, proj = interp_decomp(dS, 1e-4)

A = reconstruct_interp_matrix(idx, proj)

pass