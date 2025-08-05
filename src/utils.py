import numpy as np
#import tensorly as tl
from functools import reduce
from itertools import product

def get_dense_tensor_from_sptensor(tensor):
    shape = tensor.tensor_size
    dense_tensor = np.zeros(shape, dtype=T_train.values.dtype)
    
    coords = tensor.coords
    values = tensor.values
    
    for i in range(coords.shape[0]):
        idx = tuple(coords[i])
        dense_tensor[idx] = values[i]

    return dense_tensor

def cp_n_params(tensor_size, rnk):
    """
    Number of parameters for CP structure
    """
    return np.sum( np.array( tensor_size ) - 1) * rnk

def tucker_n_params(tensor_size, rnk):
    """
    Number of parameters for tucker structure
    """
    tensor_dim = len(tensor_size)
    n_param_core = np.prod( np.array(rnk) )
    n_param_factor = sum( tensor_size[d] * rnk[d] for d in range(tensor_dim) )
    return n_param_core + n_param_factor

def train_n_params(tensor_size, rnk):
    """
    Number of parameters for train structure
    """
    tensor_dim = len(tensor_size)
    term = 0
    for d in range(tensor_dim):
        if d == 0:
            term += tensor_size[0] * rnk[0]
        elif d == tensor_dim - 1:
            term += rnk[d-1] * tensor_size[d]
        else:
            term += rnk[d-1] * tensor_size[d] * rnk[d]
    return term

def tuple_skipping_m(N, m):
    """
    For example,
    tuple_skipping_m(5,2) = (0,1,3,4)
    tuple_skipping_m(7,3) = (0,1,2,4,5,6,7)
    tuple_skipping_m(4,1) = (0,2,3)
    """
    return tuple(i for i in range(N) if i != m)

def NL(P,T, avoid_nan=False):
    if avoid_nan:
        """
        If P has zero value, KL might be nan.
        Thus, we avoid this case
        """
        Parr = P[ P != 0 ]
        Tarr = T[ P != 0 ]
        return - np.sum(Parr * np.log(Tarr))
    else:
        return - np.sum(P * np.log(T))

def KL_div(P, T, avoid_nan=False):
    """ KL divergence from tensor P to T
    Both P and T need to be postive.
    Their total sum can be larger than 1.
    """
    if avoid_nan:
        """
        If P has zero value, KL might be nan.
        Thus, we avoid this case
        """
        Parr = P[ P != 0 ]
        Tarr = T[ P != 0 ]
        return np.sum(Parr * np.log(Parr / Tarr)) - np.sum(P) + np.sum(T)
    else:
        return np.sum(P * np.log(P / T)) - np.sum(P) + np.sum(T)

def inv_KL_div(P, T, avoid_nan=False):
    return KL_div(T, P, avoid_nan=avoid_nan)

######################
## alpha divergence ##
######################

def alpha_div_sparse(T,P,alpha,avoid_nan=False):
    if alpha == 1.0:
        return KL_div(T.values,P.values,avoid_nan=avoid_nan)

    elif alpha == 0.0:
        return inv_KL_div(T.values,P.values,avoid_nan=avoid_nan)

    else:
        tensor_size = np.size(T)
        term = np.sum( T.values**(alpha) * P.values**(1-alpha) )
        return 1.0/ ( alpha*(1-alpha) ) * ( tensor_size - term )

def alpha_div(T,P,α,avoid_nan=False):
    if α == 1.0:
        return KL_div(T,P,avoid_nan=avoid_nan)

    elif α == 0.0:
        return inv_KL_div(T,P,avoid_nan=avoid_nan)

    else:
        term1 = α * np.sum( T )
        term2 = (1-α) * np.sum( P )
        term3 = np.sum( T**α * P**(1-α) )
        return 1.0 / ( α*(1-α) ) * (term1 + term2 - term3)


def Fnorm(P, T):
    """ Frobenius norm between tensor P to T 
    Both P and T need to have same number of 
    elements.
    """
    return tl.norm(P-T)

def get_rnk_indices_for_sum(k, ik, rnk):
    """
    Get all rnk vectors whose k-th index is ik.
    Example
    get_rnk_sum_indices(0,1,[2,2,2])
    (1,0,0)
    (1,0,1)
    (1,1,0)
    (1,1,1)
    """
    rnk_dim = len(rnk)
    indices_rnk_except_k_ik = [ [ rd for rd in range(rnk[d]) ] if d != k else [ik] for d in range(rnk_dim) ]
    #for t in product(*indices_rnk_except_k_ik):
    #    print(t)
    return indices_rnk_except_k_ik


def adjust_learn_flags(model, R_cp, R_tucker, R_train, D):
    # If the target rank is 0,
    # We do not include the mixture
    # 0 rank input is priorized than model command
    learn_cp, learn_tucker, learn_train, learn_noise = model
    if R_cp == 0:
        if learn_cp:
            print("\nYou include CP model, but CP rank is 0")
            print("Thus, the CP model is removed\n")
        learn_cp = 0
    if R_tucker == 0 or R_tucker == [0 for d in range(D)]:
        if learn_tucker:
            print("\nYou include Tucker model, but Tucker rank is 0")
            print("Thus, the Tucker model is removed\n")
        learn_tucker = 0
    if R_train == 0 or R_train == [0 for d in range(D-1)] :
        if learn_tucker:
            print("\nYou include Train model, but Train rank is 0")
            print("Thus, the Train model is removed\n")
        learn_train = 0
    if R_cp == 0 and R_tucker == 0 and R_train == 0:
       raise ValueError("Chose at least one low-rank structure")

    return learn_cp, learn_tucker, learn_train, learn_noise

def adjust_ranks(Rs):
    R_cp, R_tucker, R_train = Rs
    # If the given ranks are numpy formats,
    # convert to list
    if type(R_train) == np.ndarray:
        R_train = R_train.tolist()
    if type(R_tucker) == np.ndarray:
        R_tucker = R_tucker.tolist()
    return R_cp, R_tucker, R_train

def initialize_weights(init_weights, model):
    if init_weights == None:
        eta_cp     = 1/np.sum(model)
        eta_train  = 1/np.sum(model)
        eta_tucker = 1/np.sum(model)
        eta_noise  = 1/np.sum(model)
    elif init_weights == "random":
        init_weights = np.random.rand(4)
        total_weights = sum(init_weights)
        init_weights = [ init_weights[k] / total_weights for k in range(4)]
        eta_cp, eta_tucker, eta_train, eta_noise = init_weights
    else:
        total_weights = sum(init_weights)
        if total_weights != 1.0:
            print("We normalized the given weights...")
            init_weights = [ init_weights[k] / total_weights for k in range(4)]
        eta_cp, eta_tucker, eta_train, eta_noise = init_weights
    return eta_cp, eta_train, eta_tucker, eta_noise

def print_verbose_top(alpha, N, n_para_cp, n_para_tucker, n_para_train, n_params_total, R_cp, R_tucker, R_train, 
        learn_weights, learn_cp, learn_tucker, learn_train, learn_noise, sparse=True):
    if sparse:
        print("\nEM mixture tensor learning for SPARSE data")
    else:
        print("\nEM mixture tensor learning for DENSE data") 

    objective = alpha
    if alpha == 1.0:
       objective = "KL"
    if alpha == 0.0:
        objective = "Reverse KL"

    ## Show what low-rank structure will be used.
    print("Included low-rank structures:")
    if learn_cp:
        print(f"{'CPD':<10} {'n_params'}:{n_para_cp:<8} {'Rank':<5}:{R_cp:<5}")
    if learn_tucker:
        print(f"{'Tucker':<10} {'n_params'}:{n_para_tucker:<8} {'Rank':<5}:{R_tucker}")
    if learn_train:
        print(f"{'Train':<10} {'n_params'}:{n_para_train:<8} {'Rank':<5}:{R_train}")
    if learn_noise:
        print(f"{'Noise':<10}")

    print(f"{'Learn weights':<25}:{str(learn_weights):>10}.")

    ## Show the number of parameters
    print(f"\n{'Total number of params':<25}:{n_params_total:10d}")
    print(f"{'Sample number in data':<25}:{N:10d}")
    print(f"{'Objective function':<25}:{objective:>10}-div.")

    ## Header of verbose
    if sparse:
        print(f"\n{'Iteration':<11} {'KL Error':<13} {'α-div':<13}  {'Weights':<7}  {'CP':<6}  {'Tucker':<6}  {'Train':<7} {'Noise':<9} {'Elapsed time'}")
    else:
        print(f"\n{'Iteration':<11} {'KL Error':<13} {'α-Error':<11} {'L2 Error':<14}  {'Weights':<8} {'CP':<6}  {'Tucker':<6}  {'Train':<7} {'Noise':<8}  {'Sum':<9}")
