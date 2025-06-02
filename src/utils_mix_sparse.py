import numpy as np
import math
import utils_train_sparse as setr
from itertools import product

def get_vals_from_mixture(indices, factors, model=(1,1,1,1)):
    learn_cp, learn_tucker, learn_train, learn_noise = model
    
    As = factors["As"]
    if learn_cp and As != 0:
        eta_cp = factors["eta_cp"]
        Pcp_values = get_vals_CP(indices, As, noise=0.0)
        tensor_dim = len(As[0])
        tensor_size = np.array([ len(As[0][d]) for d in range(tensor_dim) ], dtype=np.float64) 
    else:
        #print("No CP structure in this mixture")
        eta_cp = 0
        Pcp_values = 0

    G_tucker = factors["G_tucker"]
    if learn_tucker and type(G_tucker) != int:
        eta_tucker = factors["eta_tucker"]
        As_tucker = factors["As_tucker"]
        Ptucker_values = get_vals_Tucker(indices, G_tucker, As_tucker)
    
        tensor_dim = len(As_tucker)
        tensor_size = [np.shape(As_tucker[d])[0] for d in range(tensor_dim)]
        #print(tensor_dim, tensor_size)
    else:
        #print("No Tucker structure in this mixture")
        eta_tucker = 0
        Ptucker_values = 0

    G_train = factors["G_train"]
    if learn_train and type(G_train) != int:
        eta_train = factors["eta_train"]
        Ptrain_values  = get_vals_train(indices, G_train)
        
        tensor_dim = len(G_train)
        tensor_size = [ np.shape(G_train[d])[1] for d in range(tensor_dim) ]
        #print(tensor_dim, tensor_size)
    else:
        #print("No Train structure in this mixture")
        eta_train = 0
        Ptrain_values = 0

    if learn_noise:
        eta_noise = factors["eta_noise"]
        AbsOmegaI = np.prod(tensor_size)
        Pnoise = eta_noise / AbsOmegaI
    else:
        #print("No noise parameter in this mixture")
        Pnoise = 0

    P_values = eta_cp * Pcp_values + eta_tucker * Ptucker_values + eta_train * Ptrain_values + Pnoise
    return P_values


def mixture_total_sum(factors, model=(1,1,1,1)):
    """
    Get total sum of mixture tensor
    """
    
    learn_cp, learn_tucker, learn_train, learn_noise = model
    
    As = factors["As"]
    if learn_cp and As != 0:
        eta_cp = factors["eta_cp"]
        Pcp_total = CP_total_sum(As) 
        tensor_dim = len(As[0])
        tensor_size = np.array([ len(As[0][d]) for d in range(tensor_dim) ], dtype=np.float64) 
        #print(tensor_dim, tensor_size)
    else:
        print("No CP structure in this mixture")
        eta_cp = 0
        Pcp_total = 0

    G_tucker = factors["G_tucker"]
    if learn_tucker and type(G_tucker) != int:
        eta_tucker = factors["eta_tucker"]
        As_tucker = factors["As_tucker"]
        Ptucker_total = Tucker_total_sum(G_tucker, As_tucker)
    
        tensor_dim = len(As_tucker)
        tensor_size = [np.shape(As_tucker[d])[0] for d in range(tensor_dim)]
        #print(tensor_dim, tensor_size)
    else:
        print("No Tucker structure in this mixture")
        eta_tucker = 0
        Ptucker_total = 0

    G_train = factors["G_train"]
    if learn_train and type(G_train) != int:
        eta_train = factors["eta_train"]
        Ptrain_total  = train_total_sum(G_train)
        
        tensor_dim = len(G_train)
        tensor_size = [ np.shape(G_train[d])[1] for d in range(tensor_dim) ]
        #print(tensor_dim, tensor_size)
    else:
        print("No Train structure in this mixture")
        eta_train = 0
        Ptrain_total = 0

    if learn_noise:
        eta_noise = factors["eta_noise"]
        Pnoise_total = eta_noise
    else:
        print("No noise parameter in this mixture")
        Pnoise_total = 0

    P_values = eta_cp * Pcp_total + eta_tucker * Ptucker_total + eta_train * Ptrain_total + Pnoise_total
    return P_values

def mixture_to_dense(factors, model=(1,1,1,1)):
    """
    Convert mixture factors to dense tensor
    """
    
    learn_cp, learn_tucker, learn_train, learn_noise = model
    
    As = factors["As"]
    if learn_cp and As != 0:
        eta_cp = factors["eta_cp"]
        Pcp = CP_to_dense(As, noise=0.0)
        tensor_dim = len(As[0])
        tensor_size = np.array([ len(As[0][d]) for d in range(tensor_dim) ], dtype=np.float64) 
    else:
        print("No CP structure in this mixture")
        eta_cp = 0
        Pcp = 0

    G_tucker = factors["G_tucker"]
    if learn_tucker and type(G_tucker) != int:
        eta_tucker = factors["eta_tucker"]
        As_tucker = factors["As_tucker"]
        Ptucker = Tucker_to_dense(G_tucker, As_tucker)
    
        tensor_dim = len(As_tucker)
        tensor_size = [np.shape(As_tucker[d])[0] for d in range(tensor_dim)]
        print(tensor_dim, tensor_size)
    else:
        print("No Tucker structure in this mixture")
        eta_tucker = 0
        Ptucker = 0

    G_train = factors["G_train"]
    if learn_train and type(G_train) != int:
        eta_train = factors["eta_train"]
        Ptrain  = train_to_dense(G_train)
        
        tensor_dim = len(G_train)
        tensor_size = [ np.shape(G_train[d])[1] for d in range(tensor_dim) ]
        print(tensor_dim, tensor_size)
    else:
        print("No Train structure in this mixture")
        eta_train = 0
        Ptrain = 0

    if learn_noise:
        eta_noise = factors["eta_noise"]
        AbsOmegaI = np.prod(tensor_size)
        Pnoise = eta_noise / AbsOmegaI
    else:
        print("No noise parameter in this mixture")
        Pnoise = 0

    P = eta_cp * Pcp + eta_tucker * Ptucker + eta_train * Ptrain + Pnoise
    return P
    

"""
Reconst from sparse CP for dense CP
"""
def get_val_CP(idx, A, noise=0):
    """
    Get a velue on index idx of CP tensor whose factors are A
    Since factor matrices has different srtucre between dence or sparase,
    This function can be adaptable for only sparase A
    """
    rnk = len(A)
    tensor_dim = len(idx)
    assert tensor_dim == len(A[0]), "idx dim is inconsistent with factor size"
    #value_on_idx = sum( math.prod( A[r][d][ idx[d] ] for d in range(tensor_dim) ) for r in rnk)
    
    tensor_size = np.array([ len(A[0][d]) for d in range(tensor_dim) ], dtype=np.float64)
    AbsOmegaI = math.prod( tensor_size )
    
    q = np.zeros(rnk)
    for r in range(rnk):
        q[r] = math.prod( A[r][d][ idx[d] ] for d in range(tensor_dim) )
    value_on_idx = sum(q)
    return (1 - noise) * value_on_idx + noise / AbsOmegaI

def get_vals_CP(indices, A, noise=0):
    tensor_dim = len(A[0])
    tensor_size = np.array([ len(A[0][d]) for d in range(tensor_dim) ], dtype=np.float64)
    AbsOmegaI = math.prod( tensor_size )

    low_rank_values = np.zeros( len(indices) )
    for n, idx in enumerate(indices):
        low_rank_value = get_val_CP(idx, A, noise=noise)
        low_rank_values[n] = low_rank_value
        
    return low_rank_values

def CP_to_dense(A, noise=0):
    rnk = len(A)
    tensor_dim = len(A[0])
    tensor_size = np.array([ len(A[0][d]) for d in range(tensor_dim) ])
    dense_CP = np.zeros( tensor_size )
    for idx in product( *(range(Jd) for Jd in tensor_size  ) ):
        dense_CP[idx] = get_val_CP(idx, A, noise=noise)
    return dense_CP

def CP_total_sum(A, noise=0):
    rnk = len(A)
    tensor_dim = len(A[0])
    tensor_size = np.array([ len(A[0][d]) for d in range(tensor_dim) ])
    
    k = 0
    for idx in product( *(range(Jd) for Jd in tensor_size  ) ):
        k += get_val_CP(idx, A, noise=noise)
    return k

"""
Reconst from sparse Tucker to 
"""
def get_vals_Tucker(idxs, G, A, noise=0):
    values_on_idces = [ get_val_Tucker(idx, G, A, noise=noise) for idx in idxs ]
    values_on_idces  = np.array(values_on_idces)
    return values_on_idces

def get_val_Tucker(idx, G, A, noise=0):
    rnk = G.shape
    tensor_dim = len(rnk)
    R1R2R3 = list( range(Rd) for Rd in rnk )

    assert tensor_dim == len(idx), "idx dim is inconsistent with factor size"
    
    tensor_size = [ np.shape(A[d])[0] for d in range(tensor_dim) ]
    AbsOmegaI = math.prod( np.array(tensor_size, dtype=np.float64) )
    
    q = np.zeros((rnk))
    for r1r2r3 in product( *R1R2R3 ):
        q[r1r2r3] = G[r1r2r3] * math.prod( A[d][idx[d],r1r2r3[d]] for d in range(tensor_dim) )
    value_on_idx = (1 - noise ) * np.sum(q) + noise / AbsOmegaI
    return value_on_idx

def Tucker_to_dense(G, A, noise=0):
    rnk = G.shape
    tensor_dim = len(rnk)
    tensor_size = np.array([ A[d].shape[0] for d in range(tensor_dim) ])
    dense_Tucker = np.zeros( tensor_size )
    R1R2R3 = list( range(Rd) for Rd in rnk )
    I1I2I3 = list( range(Id) for Id in tensor_size ) 
    for i1i2i3 in product( *I1I2I3 ):
        dense_Tucker[i1i2i3] = get_val_Tucker(i1i2i3, G, A, noise=noise)
    return dense_Tucker

def Tucker_total_sum(G, A, noise=0):
    rnk = G.shape
    tensor_dim = len(rnk)
    tensor_size = np.array([ A[d].shape[0] for d in range(tensor_dim) ])
    #dense_Tucker = np.zeros( tensor_size )
    I1I2I3 = list( range(Id) for Id in tensor_size ) 
    k = 0
    for i1i2i3 in product(*I1I2I3):
        k += get_val_Tucker(i1i2i3, G, A)
    return k
    
"""
Reconst from sparse Train to Train 
"""

def get_vals_train(indices, G):
    coords_L = setr.get_coords_L(indices)
    GL = setr.get_sparse_train_L(coords_L, G)
    return GL[-1].values
   
def get_val_train(idx, cores, noise=0.0):
    tensor_dim = len(cores)
    rnk = [ np.shape(cores[d])[0] for d in range(1,tensor_dim) ]
    
    assert tensor_dim == len(idx), "idx dim is inconsistent with factor size"
    
    tensor_shape = [core.shape[1] for core in cores]
    AbsOmegaI = np.prod(np.array(tensor_shape, dtype=np.float64))
    
    k = 0
    for r in product(*( range(Rd) for Rd in rnk )):
        m = cores[0][0, idx[0], r[0]] * cores[tensor_dim-1][r[tensor_dim-2], idx[tensor_dim-1], 0]
        for d in range(1, tensor_dim-1):
            m *= cores[d][r[d-1], idx[d], r[d]]
        k += m
    return (1 - noise) * k + noise / AbsOmegaI

"""
def get_vals_train(idxs, G, noise=0.0):
    tensor_dim  = len(G)
    tensor_size = [ np.shape(G[d])[1] for d in range(tensor_dim) ]
    #AbsOmegaI = np.prod(np.shape(tensor_size))
    AbsOmegaI = np.prod(np.array(tensor_size, dtype=np.float64))
    
    values = [ get_val_train(idx, G, noise=noise) for idx in idxs ]
    values = np.array(values)
    return values
"""

def train_to_dense(cores, noise=0.0):
    tensor_shape = [core.shape[1] for core in cores]
    AbsOmegaI = np.prod(np.array(tensor_size, dtype=np.float64))
    
    result = np.zeros(tensor_shape)
    indices = np.indices(tensor_shape).reshape(len(tensor_shape), -1).T
    for idx in indices:
        temp = cores[0][:, idx[0], :].reshape(-1) # The first core
        for i in range(1, len(cores)):
            core = cores[i]
            temp = np.tensordot(temp, core[:, idx[i], :], axes=([0], [0]))
        result[tuple(idx)] = temp.item()
    return (1-noise) * result + noise / AbsOmegaI


def train_total_sum(cores, noise=0):
    tensor_dim  = len(cores)
    tensor_size = [ np.shape(cores[d])[1] for d in range(tensor_dim) ]
    #dense_Tucker = np.zeros( tensor_size )
    I1I2I3 = list( range(Id) for Id in tensor_size ) 
    k = 0
    for i1i2i3 in product(*I1I2I3):
        k += get_val_train(i1i2i3, cores)
    return k
