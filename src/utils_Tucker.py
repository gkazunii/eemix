import tensor_op as op
from itertools import product
from functools import reduce
import numpy as np

def Tucker_from_factors(core, factors):
    """
    Get low-tucker rank tensors from core tensor
    and factor matrices
    """
    D = tensor_dim = len(factors)
    reconst = op.mode_product(core, factors[0], 1)
    for d in range(1,D):
        reconst = op.mode_product(reconst, factors[d], d+1)
    return reconst

def get_low_Tucker_rank_tensor(rnk, tensor_size):
    tensor_dim = len(rnk)
    As = [ np.random.rand(tensor_size[d], rnk[d]) for d in range(tensor_dim) ]
    G  = np.random.rand(*rnk)
    P = Tucker_from_factors(G, As)
    return P

def get_Tucker_Q(G, As):
    tensor_dim  = len(As)
    rnk = np.shape(G)
    indices_rnk = [ [ rd for rd in range(rnk[d])] for d in range(tensor_dim)]
    Q = { k : G[k] * reduce(np.multiply.outer, [As[d][:,k[d]] for d in range(tensor_dim)]) for k in product( *indices_rnk) }
    #Q = {}
    #for k in product( *indices_rnk ):
    #    Q[k] = reduce(np.multiply.outer, [As[d][:,k[d]] for d in range(tensor_dim)])
    return Q
