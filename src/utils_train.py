import numpy as np
from itertools import product

def get_low_train_rank_tensor(R, J):
    D = len(J)
    assert len(R) + 1 == D, "rank is invalid"
    cores = [ np.array([]) for d in range(D) ]
    cores[0] = np.random.rand(1, J[0], R[0])
    for d in range(1,D-1):
        cores[d] = np.random.rand(R[d-1], J[d], R[d])
    cores[D-1] = np.random.rand(R[D-2], J[D-1], 1)

    return train_from_cores(cores)

def train_from_cores_fast(cores):
    """
    Convert a list of TT cores into a full tensor.

    Parameters:
        cores (list of np.ndarray): List of TT cores, where each core is of shape (r_{k-1}, n_k, r_k)

    Returns:
        tensor (np.ndarray): Full tensor reconstructed from the TT cores.
    """
    tensor = cores[0]  # shape (1, J0, R0)
    tensor = np.transpose(tensor, (1, 0, 2))  # shape (J0, 1, R0)

    for core in cores[1:]:
        # core shape: (R_{k-1}, J_k, R_k)
        core = np.transpose(core, (1, 0, 2))  # shape (J_k, R_{k-1}, R_k)

        # Current tensor shape: (J0, ..., J_{k-1}, R_{k-1})
        tensor_shape = tensor.shape
        tensor = tensor.reshape(-1, tensor_shape[-1])  # flatten to 2D: (prod(Js), R_{k-1})

        # Contract over R_{k-1}
        contracted = np.tensordot(tensor, core, axes=([1], [1]))  # (prod(Js), J_k, R_k)

        # Rearrange to (J0, ..., J_k, R_k)
        n_dims = len(tensor_shape) - 1
        tensor = contracted.reshape(*tensor_shape[:-1], core.shape[0], core.shape[2])
        tensor = np.moveaxis(tensor, -3, n_dims)  # move new mode to the end

    return tensor.squeeze()  # remove singleton dimensions (e.g., at the ends)

def train_from_cores(cores):
    tensor_shape = [core.shape[1] for core in cores]
    result = np.zeros(tensor_shape)
    indices = np.indices(tensor_shape).reshape(len(tensor_shape), -1).T
    for idx in indices:
        temp = cores[0][:, idx[0], :].reshape(-1) # The first core
        for i in range(1, len(cores)):
            core = cores[i]
            temp = np.tensordot(temp, core[:, idx[i], :], axes=([0], [0]))
        result[tuple(idx)] = temp.item()
    return result
    
def train_from_cores_idx(cores,idx):
    tensor_dim = len(cores)
    rnk = [ np.shape(cores[d])[0] for d in range(1,tensor_dim) ]
    k = 0
    for r in product(*( range(Rd) for Rd in rnk )):
        m = cores[0][0, idx[0], r[0]] * cores[tensor_dim-1][r[tensor_dim-2], idx[tensor_dim-1], 0]
        for d in range(1, tensor_dim-1):
            m *= cores[d][r[d-1], idx[d], r[d]]
        k += m
    return k

def get_train_R(cores):
    # G( --> d )
    # GR[d][i1,i2,...,id,rd]
    
    GR = {}
    D  = len(cores)
    GR[-1] = np.array([1])
    GR[0] = cores[0][0,:,:]
    for d in range(1,D):
        GR[d] = np.tensordot(GR[d-1], cores[d], axes=1)
        
    ## GR[D-1] should be same as full_reconst 
    ## print( np.squeeze(GR[D-1]) - train_from_cores(cores) )
    return GR

def get_train_L(cores):
    GL = {}

    # G( d <--- )
    # GL[d][rd,id+1,id+2,...,iD]
    
    D = len(cores)
    GL[D-1] = np.array([1])
    # if you do not need full_reconst, GL[-1], loop
    # for d in range(2,D+1), instead of the below code
    for d in range(2,D+2):
        GL[D-d] = np.tensordot( cores[D-d+1], GL[D-d+1], axes=[[2],[0]])
        
    #GL[0-1] should be same as full_reconst 
    # print( np.squeeze(GL[0-1]) - train_from_cores(cores) )
    
    return GL