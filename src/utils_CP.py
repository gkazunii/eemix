import numpy as np
from functools import reduce

def get_low_CP_rank_tensor(rnk, tensor_size):
    P = np.zeros(tensor_size)
    tensor_dim = np.ndim(P)
    for r in range(rnk):
        P += reduce(np.multiply.outer, [np.random.rand( tensor_size[d] ) for d in range(tensor_dim) ] )
    return P

def CP_from_factors(As):
    """
    Get low CP rank tensor from factor matrices
    """
    rnk = np.shape(As[0])[1]
    tensor_dim  = len(As)
    tensor_size = [ np.shape(As[d])[0] for d in range(tensor_dim) ]
    P = np.zeros(tensor_size)
    for r in range(rnk):
        P += reduce(np.multiply.outer, [ As[d][:,r] for d in range(tensor_dim) ] )
    return P

def CP_from_factors_(P,As):
    rnk = np.shape(As[0])[1]
    tensor_dim  = np.ndim(P)
    
    P = reduce(np.multiply.outer, [ As[d][:,0] for d in range(tensor_dim) ] )
    for r in range(1,rnk):
        P += reduce(np.multiply.outer, [ As[d][:,r] for d in range(tensor_dim) ] )
    return P

def get_CP_Q(As):
    tensor_dim  = len(As)
    rnk = np.shape(As[0])[1]
    Q = { r : reduce(np.multiply.outer, [ As[d][:,r] for d in range(tensor_dim) ] )  for r in range(rnk) }
    return Q