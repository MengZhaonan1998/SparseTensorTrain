import numpy as np
import scipy as sp
from scipy.linalg.interpolative import interp_decomp
from scipy.sparse import random

S = random(5, 5, 0.25, )
dS = S.toarray()

k, idx, proj = interp_decomp(dS, 1e-4)


pass