import utils_Tucker as Tucker
import utils_train as train
import numpy as np
import utils
import sp_tensor
from itertools import product
import importlib
importlib.reload(sp_tensor)

import sys

def get_idx_d_id(idx, tensor_dim, tensor_size):
    idx_d_id = {}
    for d in range(tensor_dim):
        for i in range(tensor_size[d]):
            idx_d_id[d,i] = idx[idx[:, d] == i].tolist()
    return idx_d_id

def get_coords_R(coords): # ( --> d )
    D = coords[0].shape[0]
    #coords_R = [ np.unique(coords[:,0:d+1],axis=0) for d in range(D) ]
    coords_R = [ coords[:,0:d+1] for d in range(D) ]
    return coords_R

def get_coords_L(coords): #( <-- d )
    D = coords[0].shape[0]
    #coords_L = [ np.unique(coords[:, D-d-1:D],axis=0) for d in range(D) ]
    coords_L = [ coords[:, D-d-1:D] for d in range(D) ]
    return coords_L

def get_sparse_train_R(coords_R, G): # ( --> d)
    D = len(G)
    tensor_size = [ np.shape(G[d])[1] for d in range(D) ]
    R = [ np.shape(G[d])[2] for d in range(D) ]

    GR = {}
    ## d = -1
    GR[-1] = 1
    
    GR[0]  = sp_tensor.dense_to_sparse(np.squeeze(G[0], 0))
    for d in range(1,D):
        coords_Rd  = coords_R[d]
        GRd_coords = [ [idx for idx in coords_rd] + [rd] for coords_rd in coords_Rd for rd in range(R[d]) ]
        GRd_values = [ sum( GR[d-1].coord_to_value[ *idxr[0:d] + [rdm1] ] * G[d][rdm1, idxr[-2], idxr[-1]] for rdm1 in range(R[d-1]) ) for idxr in GRd_coords ]
        tensor_size_GRd = tensor_size[0:d+1] + [R[d]]
        GR[d] = sp_tensor.Sp_tensor( np.array(GRd_coords), np.array(GRd_values), tensor_size_GRd, check_empty=False )

    # GR[D-1] will be the same tensor as the reconst. 
    # i.e., GR[D-1] == train_from_cores(G)
    return GR

def get_sparse_train_L(coords_L, G): #( d <-- )
    D = len(G)
    tensor_size = [ np.shape(G[d])[1] for d in range(D) ]
    R = [ np.shape(G[d])[2] for d in range(D) ]

    GL = {}
    
    # d = D-1
    GL[D-1] = np.array([1])

    # d = D-2
    coords_Ld  = coords_L[0]
    GLd_coords = [ [rd] + [idx for idx in coords_ld] for coords_ld in coords_Ld for rd in range(R[-2]) ]
    GLd_values = [ sum( G[D-1][ridx[0], ridx[1] ,rd] for rd in range(R[-1]) ) for ridx in GLd_coords ]
    tensor_size_GLd = [ R[-2] ] + tensor_size[-1:] 
    GL[D-2] = sp_tensor.Sp_tensor( np.array(GLd_coords), np.array(GLd_values), tensor_size_GLd, check_empty=False )

    # d = D-3, D-4, ..., 0, -1
    for k in range(1, D):
        coords_Ld  = coords_L[k]
        if k != D - 1:
            GLd_coords = [ [rd] + [idx for idx in coords_ld] for coords_ld in coords_Ld for rd in range(R[-k-2]) ]
            tensor_size_GLd = [ R[-k-2] ] + tensor_size[-k-1:] 
        elif k == D - 1:
            # R[-D-1] is undefined. Thus, I replace R[-D-1] with 1. 
            GLd_coords = [ [rd] + [idx for idx in coords_ld] for coords_ld in coords_Ld for rd in range(1) ]
            tensor_size_GLd = [ 1 ] + tensor_size[-k-1:] 
            
        GLd_values = [ sum( G[D-k-1][ridx[0], ridx[1] ,rd] * GL[D-k-1].coord_to_value[ *([rd] + ridx[-k:]) ] for rd in range(R[-k-1]) ) for ridx in GLd_coords ]
        GL[D-k-2] = sp_tensor.Sp_tensor( np.array(GLd_coords), np.array(GLd_values), tensor_size_GLd, check_empty=False )
            
    ## For debug
    ## GL[D-2] is equivalent with G[D-1]
    ## GL[D-3] is np.tensordot( G[D-2], G[D-1],  axes=1 )
    ## GL[D-4] is np.tensordot( G[D-3], GL[D-3], axes=1 )
    ## GL[D-5] is np.tensordot( G[D-4], GL[D-4], axes=1 )
    ## ...
    ## GL[-1] is the same tensor as the reconst. 
    ## i.e., GR[-1] == train_from_cores(G)
   
    return GL

def sparse_train_reconst(G, noise, coords):
    coords_L = get_coords_L(coords)
    GL = get_sparse_train_L(coords_L,G)
    tensor_size = [ np.shape(G[d])[1] for d in range( len(G) ) ]
    AbsOmegaI = np.prod(np.array(tensor_size, dtype=np.float64))
    reconst = GL[-1]
    reconst_with_noise = (1-noise) * reconst.values + noise / AbsOmegaI
    return reconst_with_noise

def sparse_train_from_cores_with_noise(G, noise):
    values_with_noise = sparse_train_from_cores_with_noise(G, noise, idxs)
    return np.sum(total)
    

def sparse_train_from_cores_with_noise(G, noise, idxs):
    tensor_dim  = len(G)
    tensor_size = [ np.shape(G[d])[1] for d in range(tensor_dim) ]
    #AbsOmegaI = np.prod(np.shape(tensor_size))
    AbsOmegaI = np.prod(np.array(tensor_size, dtype=np.float64))
    
    values = [ train.train_from_cores_idx(G, idx) for idx in idxs ]
    values_with_noise = (1-noise) * np.array(values) + noise / AbsOmegaI
    return values_with_noise
    
