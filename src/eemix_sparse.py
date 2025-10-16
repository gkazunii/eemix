import numpy as np
import utils
import math
import os
import time

import utils_sum as us
import utils_mix_sparse as ums
from itertools import product
import sys

import utils_train_sparse as setr
import sp_tensor
import importlib
importlib.reload(us)
importlib.reload(ums)
importlib.reload(sp_tensor)

def eemix_sparse(T, Rs, alpha=1.0, model=[1,1,1,1],
        max_iter=10, iter_inside=1, update_rule=0, tol=1.0e-4,
        init_weights=None, learn_weights=True,
        init_cp="random", init_tucker="random",init_train="random",
        verbose=True, verbose_interval=1, loss_history=True, conv_check_interval=10,
        check_sum=False, avoid_nan=True):
    """
    Args:
        T (sp_tensor): sparse tensor
        Rs (list): Ranks Rs = [Rcp (int), Rtucker(list), Rtrain(list)]
        alpha (real number): alpha of alpha-divergence. 
            If alpha = 1.0, then KL div.
            If alpha = 0.5, then Hellinger distance.
        iter_inside(int>0): the number of loop in inside EM-algorithm.
        learn_weights(Boolen): True for learn mixture ratio otherwise False
        init_weights: inital values of weights (eta)
        init_cp: inital values of cp-factors A
        init_tucker: inital values of tucker core G and factors A
        init_train: inital values of train cores G
        check_sum(Boolen): just for debug.

    Returns:
        As : CP factors
        Gtucker : Core tensor of tucker
        Astucker : factor matrices of 
        G : core tensors of trains
        weights(list)

    """

    ## To obtain time each iteration
    start_time = time.perf_counter()

    # input tensor need to be normalized
    T.normalize()

    # tensor shape
    J = T.tensor_size

    # tensor dim
    D = T.tensor_dim

    # The number of samples in tensor T
    N = T.nnz

    # the size of sample space
    AbsOmegaI = math.prod(J)

    assert sum(model) != 0, "Chose at least one low-rank structure"
    assert iter_inside > 0, "iter inside need to be largaer than 1"

    # Rcp ... CP rank
    # Rtucker ... Tucker rank
    # R ... train rank
    R_cp, R_tucker, R_train = utils.adjust_ranks(Rs)

    # Flags for low-rank structures
    learn_cp, learn_tucker, learn_train, learn_noise = utils.adjust_learn_flags(model, R_cp, R_tucker, R_train, D)

    ####################
    ## Initialization ##
    ####################

    # Normalized Weight
    # NOTE: the total sum of eta should be 1.0
    model = [learn_cp, learn_tucker, learn_train, learn_noise]
    eta_cp, eta_train, eta_tucker, eta_noise = utils.initialize_weights(init_weights, model)
    ranks = {}

    # Mixture Tensor
    P = sp_tensor.Sp_tensor( T.coords, np.random.rand(N), J, normalize=True,  check_empty=True)

    # For the first E-step
    F = sp_tensor.Sp_tensor( T.coords, np.zeros(N),       J, normalize=False, check_empty=True)

    # Pure low-rank tensors
    Pcp     = sp_tensor.Sp_tensor( T.coords, np.random.rand(N), J, normalize=True )
    Ptucker = sp_tensor.Sp_tensor( T.coords, np.random.rand(N), J, normalize=True )
    Ptrain  = sp_tensor.Sp_tensor( T.coords, np.random.rand(N), J, normalize=True )

    # Init for CP factors
    if learn_cp:
        # Check cp rank condtion
        assert type(R_cp) == int, "CP rank need to be int."
        ranks["cp"] = R_cp

        Qcp = { r : sp_tensor.Sp_tensor(T.coords, np.random.rand(N), J, check_empty=False) for r in range(R_cp) }
        Mcp = { r : sp_tensor.Sp_tensor(T.coords, Qcp[r].values * T.values / P.values, J, check_empty=False) for r in range(R_cp) }
        Mcpr_sums = np.zeros(R_cp)
        if init_cp == "random":
            A = { r : [] for r in range(R_cp) } # Dense vectors
        else:
            A = init_cp
            for r in range(R_cp):
                for n in range(N):
                    Qcp[r].values[n] = math.prod( A[r][d][ T.coords[n][d] ] for d in range(D) )

            Pcp.values = sum( Qcp[r].values for r in range(R_cp) )

        # +1 because of the weight
        n_para_cp = utils.cp_n_params(J, R_cp) + 1
    else:
        eta_cp = 0.0
        A = 0
        Ccp = 0
        n_para_cp = 0

    # Init for Tuckers 
    if learn_tucker:
        # Check tucker rank condtion
        if type(R_tucker) == int:
            print("\nRtucker need to be vector")
            R_tucker = [ R_tucker for d in range(D) ]
            print(f"Thus, R_tucker is modified as a vector {R_tucker}")
        else:
            assert len(R_tucker) == D, f"Wrong dim of tucker rank. It should be {D} dim vec."

        ranks["tucker"] = R_tucker

        R1R2R3 = list( range(Rd) for Rd in R_tucker )

        if init_tucker == "random":
            A_tucker = [  np.random.rand( J[d], R_tucker[d] ) for d in range(D) ] # Dense matrices
            G_tucker = np.random.rand( *R_tucker )
        else:
            A_tucker = init_tucker[1]
            G_tucker = init_tucker[0]
            assert np.shape(G_tucker) == tuple(R_tucker), "the size of given tucker core mismatchs"
            for d in range(D):
                assert np.shape(A_tucker[d]) == (J[d], R_tucker[d]), "the size of given tucker factor mismatch"

        Qtucker = { r1r2r3 : sp_tensor.Sp_tensor(T.coords, np.random.rand(N), J, check_empty=False) \
             for r1r2r3 in product( *R1R2R3 ) }
        # update Q
        # Q = update_Tucker_Q_sparse(Q, G, A, T)
        for n in range(N):
            for r1r2r3 in product( *R1R2R3 ):
                # Naivly, G and A are dense, so Q can be also dense. 
                # However, we need only Q on T.coords. 
                # Thus, we keep Q as sparse tensor.
                Qtucker[r1r2r3].values[n] = G_tucker[r1r2r3] * math.prod( A_tucker[d][T.coords[n][d], r1r2r3[d]] for d in range(D))

        Ptucker.values = sum( Qtucker[r1r2r3].values for r1r2r3 in product( *R1R2R3) )
        Mtucker = { r1r2r3 : sp_tensor.Sp_tensor(T.coords, Qtucker[r1r2r3].values * T.values / P.values, J, check_empty=False) \
             for r1r2r3 in product( *R1R2R3 ) } 
        
        sumsM_results = { r1r2r3 : { d : [] for d in range(D) } for r1r2r3 in product( *R1R2R3 ) }
        
        # +1 because of the weight
        n_para_tucker = utils.tucker_n_params(J, R_tucker) + 1
    else:
        eta_tucker = 0.0
        G_tucker = 0
        A_tucker = 0
        Ctucker = 0
        n_para_tucker = 0

    # Init for tensor train
    if learn_train:
        # Check train rank condtion
        if type(R_train) == int:
            print("\nRtrain need to be vector")
            R_train = [ R_train for d in range(D-1) ]
            print(f"Thus, Rtrain is modified as a vector {R_train}")
            ranks["train"] = R_train
            
        else:
            assert len(R_train) == D-1, f"Wrong dim of train rank. It should be {D-1} dim vec."
        
        # Train cores
        if init_train == "random":
            G = [ np.array([]) for d in range(D) ]
            G[0] = np.random.rand(1, J[0], R_train[0])
            for d in range(1,D-1):
                G[d] = np.random.rand(R_train[d-1], J[d], R_train[d])
            G[D-1] = np.random.rand(R_train[D-2], J[D-1], 1)
        else:
            G = init_train
            assert len(G) == D, "the number of given core mismatch"
            assert np.shape(G[0]) == (1, J[0], R_train[0]), "the size of given train cores mismatch"
            for d in range(1, D-1):
                assert np.shape(G[d]) == (R_train[d-1], J[d], R_train[d]), "the size of given train cores mismatch"
            assert np.shape(G[D-1]) == (R_train[D-2], J[D-1], 1), "the size of given train cores mismatch"

        
        coords_L = setr.get_coords_L(T.coords)
        coords_R = setr.get_coords_R(T.coords)
        GR = setr.get_sparse_train_R(coords_R, G)
        GL = setr.get_sparse_train_L(coords_L, G)
        #Ptrain = GL[-1]
        Ptrain.values = GL[-1].values
    
        ## For train:
        ## Get coord where d th idx is id
        ## G is obtaind by summation of GR, G, GL on these idices.
        idx_d_id = setr.get_idx_d_id(T.coords, D, J)

        # +1 because of the weight
        n_para_train = utils.train_n_params(J, R_train) + 1
        
    else:
        n_para_train = 0
        eta_train = 0.0
        Ctrain = 0
        G = 0

    n_params_total = n_para_cp + n_para_tucker + n_para_train
    n_params_dict  = {"cp":n_para_cp, "tucker":n_para_tucker, "train":n_para_train, "total":n_params_total}

    if not(learn_noise):
        eta_noise = 0.0
        Cnoise = 0.0
    else:
        ## +1 because of the noise parameter
        n_params_total += 1

    P.values = eta_cp * Pcp.values + eta_tucker * Ptucker.values + eta_train * Ptrain.values + eta_noise / AbsOmegaI
    F.values = P.values**(1-alpha) / ( np.sum( T.values**(alpha) * P.values**(1-alpha) ) )
    T_over_P = sp_tensor.Sp_tensor( T.coords,  (T.values)**(alpha) * F.values / P.values, J)

    alpha_error = np.inf
    prev_error = np.inf
    prev_error_for_conv = np.inf

    ###############
    ## Histories ##
    ###############

    loss_kl_history  = []
    loss_nl_history  = []
    loss_alpha_history  = []
    loss_fro_history = []
    iter_history = []
    elapsed_times = []

    #############
    ## Verbose ##
    #############

    if verbose:
        utils.print_verbose_top(alpha, N,
                n_para_cp, n_para_tucker, n_para_train, n_params_total,
                R_cp, R_tucker, R_train,
                learn_weights, learn_cp, learn_tucker, learn_train, learn_noise, sparse=True)

    n_iter = 0

    ############################
    ############################
    ## Proposed EEM-algorithm ##
    ############################
    ############################

    while(n_iter < max_iter):

        ##################
        ## First E-STEP ##
        ##################
        F.values = P.values**(1-alpha) / np.sum( T.values**(alpha) * P.values**(1-alpha) )

        for _ in range(iter_inside):

            #########################
            ## EM-algorithm inside ##
            #########################
            n_iter += 1

            if learn_cp:

                ###########################
                ## M Step for CP
                ###########################
                for r in range(R_cp):
                    # Solution of many-body approximation is invarient to
                    # constant mulitplicaiton. Thus, Mcp * eta_cp or Mcp is not matter.
                    # alpha = 1.0 then
                    # Mcp[r].values = T.values * Qcp[r].values / P.values
                    Mcp[r].values = T.values**(alpha) * F.values * Qcp[r].values / P.values

                    #Mcp[r].values = eta_cp * T.values * Qcp[r].values / P.values
                    Mcpr_sums[r] = np.sum(Mcp[r].values)
                total = np.sum(Mcpr_sums)
        
                # update A
                # A[:][d][:] is dense matrix
                # A[r][d][id] where
                # r is rank, r=1,2,...,R, [rnk] 
                # d is tensor modes, d=1,2,...,D, [tensor_dim] 
                # id is d-th index of the tensor, id=1,2,..,Id [tensor_size[d]]
                for r in range(R_cp):
                    # Update by the closed-form update rule
                    sums_results = us.reduce_sum_each_dim(Mcp[r].coords, Mcp[r].values, D)
                    A[r] = [ sums_results[d][1] * (Mcpr_sums[r])**(1/D-1) * ( total ** (-1/D) ) for d in range(D) ]
                    # We can ignore total term because it will be multiplicated D-times.
                    # A[r] = [ sums_results[d][1] * (Mcpr_sums[r])**(1/D-1) for d in range(D) ]
        
                ## Mcp has no guranteed to be normalized 1.
                ## However, Pcp need to be normalize. Thus we normalize each A
                ## Normalize A[r]
                ## for r in range(R_cp):
                ##   A[r] /= total**(1/D)
        
                # Checking the normalization
                # print( secp.sparse_CP_total_sum(A) )
        
                # update Q
                for r in range(R_cp):
                    for n in range(N):
                        Qcp[r].values[n] = math.prod( A[r][d][ T.coords[n][d] ] for d in range(D) )
        
                # update low-cp tensor
                Pcp.values = sum( Qcp[r].values for r in range(R_cp) )

            if learn_tucker:

                ###########################
                ## M Step for Tucker 
                ###########################

                # update M
                for r1r2r3 in product( *R1R2R3 ):
                    # Solution of many-body approximation is invarient to
                    # constant mulitplicaiton. Thus, Mtucker * eta_tucker or Mtucker is not matter.
                    Mtucker[*r1r2r3].values = Qtucker[r1r2r3].values * (T.values)**(alpha) * F.values / P.values

                # update G
                for r1r2r3 in product( *R1R2R3 ):
                    G_tucker[*r1r2r3] = sum(Mtucker[r1r2r3].values)
                # normalize G
                G_tucker /= np.sum(G_tucker)

                # update A
                # A[d] is dense matrix
                # A[d][id,rd] where
                # rd is d-th rank, rd=1,2,...,Rd, [rnk] 
                # d is tensor modes, d=1,2,...,D, [tensor_dim] 
                # id is d-th index of the tensor, id=1,2,..,Id [tensor_size[d]]
                for r1r2r3 in product( *R1R2R3 ):
                    tmp_results = us.reduce_sum_each_dim(Mtucker[r1r2r3].coords, Mtucker[r1r2r3].values, D, sort=True)
                    for d in range(D):
                        sumsM_results[r1r2r3][d] = tmp_results[d][1]

                for d in range(D):
                    for rd in range(R_tucker[d]):
                        indices_rnk = utils.get_rnk_indices_for_sum(d, rd, R_tucker)
                        A_tucker[d][:,rd]  = sum( sumsM_results[r1r2r3][d] for r1r2r3 in product(*indices_rnk) )

                    # normalize A
                    for rd in range(R_tucker[d]):
                         A_tucker[d][:,rd] /= np.sum( A_tucker[d][:,rd] ) 

                # update Q
                # Q = update_Tucker_Q_sparse(Q, G, A, T)
                for n in range(N):
                    for r1r2r3 in product( *R1R2R3 ):
                        # Naivly, G and A are dense, so Q can be also dense. 
                        # However, we need only Q on T.coords. 
                        # Thus, we keep Q as sparse tensor.
                        Qtucker[r1r2r3].values[n] = G_tucker[r1r2r3] * math.prod( A_tucker[d][T.coords[n][d], r1r2r3[d]] for d in range(D))

                Ptucker.values = sum( Qtucker[r1r2r3].values for r1r2r3 in product( *R1R2R3) )
                
            if learn_train:
                
                ###########################
                ## M Step for Train
                ###########################
        
                ## update cores
                for d in range(D):
                    # NOTE: in the following, constant `eta_train` is in each sum 
                    # Solution of many-body approximation is invarient to
                    # constant mulitplicaiton. Thus, including eta_train 
                    # or not is not matter.
                    if d == 0:
                        # Since GR[-1] is not sparse tensor but sclaer value "1", 
                        # we need exceptional procedure as follows:
                        for rdm1, jd, rd in product( range(1), range(J[d]), range(R_train[d])):
                            G[d][rdm1, jd, rd] = \
                            sum( eta_train * T_over_P.coord_to_value[ *idx ] \
                                  * 1 \
                                  * G[d][rdm1, jd, rd] \
                                  * GL[d].coord_to_value[ *([rd] + idx[d+1:] ) ] \
                                  for idx in idx_d_id[d,jd] )
                            
                    elif d == D-1:
                         # Since GL[D-1] is not sparse tensor but sclaer value "1", 
                         # we need exceptional procedure as follows:
                         for rdm1, jd, rd in product( range(R_train[d-1]), range(J[d]), range(1) ) :
                            G[d][rdm1, jd, rd] = \
                            sum( eta_train * T_over_P.coord_to_value[ *idx ] \
                                 * GR[d-1].coord_to_value[ *(idx[0:d] + [rdm1]) ] \
                                 * G[d][rdm1, jd, rd] \
                                 * 1 \
                                 for idx in idx_d_id[d,jd] )
                             
                    else: # d = 1, 2, ..., D-2
                        for rdm1, jd, rd in product( range(R_train[d-1]), range(J[d]), range(R_train[d]) ) :
                            G[d][rdm1, jd, rd] = \
                            sum( eta_train * T_over_P.coord_to_value[ *idx ] \
                                 * GR[d-1].coord_to_value[ *(idx[0:d] + [rdm1]) ] \
                                 * G[d][rdm1, jd, rd] \
                                 * GL[d].coord_to_value[ *([rd] + idx[d+1:] ) ] \
                                 for idx in idx_d_id[d,jd] )
                
                ## Normalizer G
                for d in range(D):
                    if d != D - 1:
                        for rd in range(R_train[d]):
                            G[d][:,:,rd] /= np.sum( G[d][:,:,rd] )
                    else:
                        G[d][:,:,0] /= np.sum( G[d][:,:,0] )
        
                
                # To check if the normalization is satisified
                # print( np.sum( ut.train_from_cores(G) ) )
        
                GL = setr.get_sparse_train_L(coords_L, G)
                GR = setr.get_sparse_train_R(coords_R, G)
        
                # update low-train tensor
                Ptrain.values = GL[-1].values
                
            ###########################
            ## M Step for Weights 
            ###########################

            if learn_weights:
                if update_rule == 0:
                    if learn_cp:
                        Ccp    = eta_cp    * np.sum( T_over_P.values * Pcp.values )
                    if learn_tucker:
                        Ctucker = eta_tucker * np.sum( T_over_P.values * Ptucker.values )
                    if learn_train:
                        Ctrain = eta_train * np.sum( T_over_P.values * Ptrain.values )
                    if learn_noise:
                        Cnoise = eta_noise / AbsOmegaI * np.sum( T_over_P.values )
                else:
                    if learn_cp:
                        Ccp    = np.sum( T_over_P.values * Pcp.values )
                    if learn_tucker:
                        Ctucker = np.sum( T_over_P.values * Ptucker.values )
                    if learn_train:
                        Ctrain = np.sum( T_over_P.values * Ptrain.values )
                    if learn_noise:
                        Cnoise = 1.0 / AbsOmegaI * np.sum( T_over_P.values )

                eta_cp     = Ccp     / (Ccp + Ctucker + Ctrain + Cnoise)
                eta_tucker = Ctucker / (Ccp + Ctucker + Ctrain + Cnoise)
                eta_train  = Ctrain  / (Ccp + Ctucker + Ctrain + Cnoise)
                eta_noise  = Cnoise  / (Ccp + Ctucker + Ctrain + Cnoise)
            
            ## E-Step
            ##########################
            # update mixture tensor
            ##########################
            P.values = eta_cp * Pcp.values + eta_tucker * Ptucker.values + eta_train * Ptrain.values + eta_noise / AbsOmegaI

            ## For debug...
            ## Check normalizeing condition 
            if check_sum == True:
            # Pcp_sum = CP_total_sum(A, noise=0.0)
            # Ptucker_sum = Tucker_total_sum(G_tucker, A_tucker, noise=0.0)
            # Ptrain_sum  = train_total_sum(G)
            # print(Pcp_sum, Ptucker_sum, Ptrain_sum) ## All of them need to be 1.0
                factors = {"As":A, "G_tucker":G_tucker, "As_tucker":A_tucker, "G_train":G, "eta_cp":eta_cp, "eta_tucker":eta_tucker, "eta_train":eta_train, "eta_noise":eta_noise}
                totalsum = ums.mixture_total_sum(factors) # it is alwasy 1 becase of normalizeation
                print("Total sum:", totalsum)
            

            #T_over_P = sp_tensor.Sp_tensor( T.coords,  T.values / P.values, J)
            T_over_P.values = (T.values)**alpha * F.values / P.values
            T_over_P.update_coord_to_value()
            # the above line is required 
            # since we refer to T_over_P.coord_to_value[ *idx ] to update train


            if verbose and n_iter > 0:
                if n_iter % verbose_interval == 0:
                    # Since both P and T are normalized, 
                    # NL is also monotonically decreasing.
                    nl_error = utils.NL(T.values, P.values)
                    kl_error = utils.KL_div(T.values, P.values, avoid_nan=avoid_nan)
                    alpha_error = utils.alpha_div_sparse(T, P, alpha, avoid_nan=avoid_nan)
                    # getting fro error each step is heavy.
                    # Thus, we do not get fro error here
                    # f_error = 0
                    
                    elapsed_time = time.perf_counter() - start_time
                    print(f"Iter: {n_iter:5d} KL: {kl_error:.7f} α :{alpha_error:.7f} | Weights: {eta_cp:.5f} {eta_tucker:.5f} {eta_train:.5f} {eta_noise:.5f} | {elapsed_time:.2f} sec.")
                    if prev_error < alpha_error:
                        print("alpha-div. error is not monotonically decreasing...")
                    prev_error = alpha_error
                    
                    if loss_history:
                        loss_kl_history.append(kl_error)
                        loss_nl_history.append(nl_error)
                        loss_alpha_history.append(alpha_error)
                        #loss_fro_history.append(f_error)
                        iter_history.append(n_iter)
                        elapsed_times.append(elapsed_time)


            if n_iter > 3 and n_iter % conv_check_interval == 1 and n_iter > verbose_interval:
                kl_error = utils.KL_div(T.values, P.values, avoid_nan=avoid_nan) 
                res = abs( prev_error_for_conv - alpha_error ) / conv_check_interval  
                if res < tol:
                    break
                else:
                    prev_error_for_conv = alpha_error

    #################
    ## For outputs ##
    #################

    hisotry = {"iter":iter_history, "kl":loss_kl_history, "nl":loss_nl_history, 
                    "fro":"Not Computed", "alpha":loss_alpha_history, "time":elapsed_times}
    factors = {"As":A, "G_tucker":G_tucker, "As_tucker":A_tucker, "G_train":G, 
                    "eta_cp":eta_cp, "eta_tucker":eta_tucker, "eta_train":eta_train, "eta_noise":eta_noise}
    models  = [learn_cp, learn_tucker, learn_train, learn_noise]
    details = {"max_iter":max_iter, "n_iter":n_iter, "tol":tol, "verbose_interval":verbose_interval, 
            "n_params":n_params_dict,
            "conv_check_interval":conv_check_interval, "models":models, "rank":ranks, 
            "update_rule":update_rule, "alpha":alpha}

    ## If you wanna reconst tensor P, from factors, run..
    # import utils_mix_sparse as ums
    # ums.get_vals_from_mixture(indices, factors)
    # ums.mixsture_total_sum(factors) # it is alwasy 1 becase of normalizeation
    # ums.mixsture_to_dense(factors)  # get full values 
    return factors, P, hisotry, details

def sparse_Tucker_from_GA_values(G, A, idxs):
    values_on_idces = [ sparse_Tucker_from_GA(G, A, idx) for idx in idxs ]
    return values_on_idces

def sparse_Tucker_from_GA(G, A, idx):
    rnk = G.shape
    tensor_dim = len(rnk)
    R1R2R3 = list( range(Rd) for Rd in rnk )
    q = np.zeros((rnk))
    for r1r2r3 in product( *R1R2R3 ):
        q[r1r2r3] = G[r1r2r3] * math.prod( A[d][idx[d],r1r2r3[d]] for d in range(tensor_dim) )
    value_on_idx = np.sum(q)
    return value_on_idx


