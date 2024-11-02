import numpy as np
import tensorly as tl
from STTSVD import ttsvd

def contract(tensor1, modes1, tensor2, modes2):
    """Tensor contraction between two tensors on specified modes
    
    Parameters
    ----------
    tensor1 : tl.tensor
    modes1 : int list or int
        modes on which to contract tensor1
    tensor2 : tl.tensor
    modes2 : int list or int
        modes on which to contract tensor2

    Returns
    -------
    contraction : tensor1 contracted with tensor2 on the specified modes
    """
    if isinstance(modes1, int):
        modes1 = [modes1]
    if isinstance(modes2, int):
        modes2 = [modes2]
    modes1 = list(modes1)
    modes2 = list(modes2)
    
    if len(modes1) != len(modes2):
        raise ValueError('Can only contract two tensors along the same number of modes'
                         '(len(modes1) == len(modes2))'
                         'However, got {} modes for tensor 1 and {} mode for tensor 2'
                         '(modes1={}, and modes2={})'.format(
                           len(modes1), len(modes2), modes1, modes2))
    
    contraction_dims = [tl.shape(tensor1)[i] for i in modes1]
    if contraction_dims != [tl.shape(tensor2)[i] for i in modes2]:
        raise ValueError('Trying to contract tensors over modes of different sizes'
                         '(contracting modes of sizes {} and {}'.format(
                             contraction_dims, [tl.shape(tensor2)[i] for i in modes2]))
    shared_dim = int(np.prod(contraction_dims))

    modes1_free = [i for i in range(tl.ndim(tensor1)) if i not in modes1]
    free_shape1 = [tl.shape(tensor1)[i] for i in modes1_free]

    tensor1 = tl.reshape(tl.transpose(tensor1, modes1_free + modes1),
                         (int(np.prod(free_shape1)), shared_dim))
    
    modes2_free = [i for i in range(tl.ndim(tensor2)) if i not in modes2]
    free_shape2 = [tl.shape(tensor2)[i] for i in modes2_free]

    tensor2 = tl.reshape(tl.transpose(tensor2, modes2 + modes2_free),
                         (shared_dim, int(np.prod(free_shape2))))
    
    res = tl.dot(tensor1, tensor2)
    return tl.reshape(res, tuple(free_shape1 + free_shape2))



A = tl.zeros([2,4,3])
for i in range(2):
    for j in range(4):
        for k in range(3):
            A[i,j,k] = i + j + k 

B = tl.zeros([3,4,5])
for i in range(3):
    for j in range(4):
        for k in range(5):
            B[i,j,k] = i+j-k

#C = contract(A,[1,2], B, [1,0])
#D = tl.reshape(A, [8,3])

x = ttsvd(A, 5, 1e-5)

pass
