import numpy as np
import scipy.linalg as la
import scipy.linalg.interpolative as sli
import matplotlib.pyplot as plt


#from scipy.sparse import random
#from scipy.linalg.interpolative import interp_decomp
#from scipy.linalg.interpolative import reconstruct_interp_matrix

n = 100
A0 = np.random.randn(n, n)

density = 0.2
for i in range(n):
    for j in range(n):
        thres = np.random.rand()
        if thres > density:
            A0[i,j] = 0.0

U0, sigma0, VT0 = la.svd(A0)
print(la.norm((U0*sigma0).dot(VT0) - A0))

sigma = np.exp(-np.arange(n))

A = (U0 * sigma).dot(VT0)
A = A0

plt.figure()
plt.semilogy(sigma)
plt.show()

k = 50
idx, proj = sli.interp_decomp(A, k)
print(idx)

B = A[:, idx[:k]]
P = np.hstack([np.eye(k), proj])[:, np.argsort(idx)]
Aapprox = np.dot(B, P)
print(la.norm(A - Aapprox, 2)/la.norm(A, 2))

plt.figure()
plt.imshow(np.hstack([np.eye(k), proj]))
plt.show()
pass