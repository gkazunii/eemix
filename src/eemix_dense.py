import sys
import os
import numpy as np
import time
from itertools import product

import utils
import utils_train as train
import utils_CP as CP
import utils_Tucker as Tucker
import eemix_sparse

import importlib
importlib.reload(train)
importlib.reload(utils)
importlib.reload(eemix_sparse)

def eemix(T, Rs, alpha=1.0, model=[1,1,1,1],
        max_iter=10, iter_inside=1, update_rule=0, tol=1.0e-4,
        init_weights=None, learn_weights=True,
        verbose=True, verbose_interval=1, loss_history=True, conv_check_interval=10,
        avoid_nan=True):
    """
    Args:
        T  (multidimensional array): input tensor
        Rs (list): Ranks Rs = [Rcp (int), Rtucker(list), Rtrain(list)]
        alpha (real number): alpha of alpha-divergence.
            If alpha = 1.0, then KL div.
            If alpha = 0.5, then Hellinger distance.
        iter_inside(int>0): the number of loop in inside EM-algorithm.
        learn_weights(Boolen): True for learn mixture ratio otherwise False
        loss_history(Boolen): True to record loss history otherwise False
        avoid_nan(Boolen): True to ignore zero value in T and avoid log(0.0)
        init_weights(4-dim list): initial weights. None for random init

    Returns:
        factors(dict): obtained factors
        P (multidimensional array): mixtured low-rank tensor
        history: loss curve
        details: running details
    """

    ## To obtain time each iteration
    start_time = time.perf_counter()

    # Normalize input tensor
    T = T / np.sum(T)

    # tensor dim
    D = np.ndim(T)

    # tensor shape
    J = np.shape(T)

    # For the first E of EEM steps
    F = np.zeros(J)

    # the size of sample space
    AbsOmegaI = np.prod(J)

    assert sum(model) != 0, "Chose at least one low-rank structure"
    assert iter_inside > 0, "iter inside need to be largaer than 1"

    ## Rcp ... CP rank
    ## Rtucker ... Tucker rank
    ## R ... train rank
    Rcp, Rtucker, R = utils.adjust_ranks(Rs)

    # Flags for low-rank structures
    learn_cp, learn_tucker, learn_train, learn_noise = utils.adjust_learn_flags(model, Rcp, Rtucker, R, D)

    ####################
    ## Initialization ##
    ####################

    # Normalized Weight
    # NOTE: the total sum of eta should be 1.0
    model = [learn_cp, learn_tucker, learn_train, learn_noise]
    eta_cp, eta_train, eta_tucker, eta_noise = utils.initialize_weights(init_weights, model)

    """
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
    """


    ## Init for EMTrain
    ranks = {}
    if learn_train:
        if type(R) == int:
            print("\nRtrain need to be vector")
            R = [ R for d in range(D-1) ]
            print(f"Thus, Rtrain is modified as a vector {R}")
        else:
            assert len(R) == D-1, "Wrong dim of train rank"

        G = [ np.array([]) for d in range(D) ]

        G[0] = np.random.rand(1, J[0], R[0])
        for d in range(1,D-1):
            G[d] = np.random.rand(R[d-1], J[d], R[d])
        G[D-1] = np.random.rand(R[D-2], J[D-1], 1)

        # normalized core_tensors
        G[0] = G[0] / G[0].sum(axis=0, keepdims=True) / J[0]
        for d in range(1,D-1):
            G[d] = G[d] / G[d].sum(axis=(0,2), keepdims=True) / J[d]
        G[D-1] = G[D-1] / G[D-1].sum(axis=2, keepdims=True) / J[D-1]

        Ptrain = train.train_from_cores_fast(G)
        r = [*R, 1]
        ranks["train"] = R

        # +1 because of the weight
        n_para_train = utils.train_n_params(J, R) + 1
    else:
        Ptrain = 0
        Ctrain = 0
        eta_train = 0
        G = 0
        n_para_train = 0

    ## Init for EMCP
    if learn_cp:
        assert type(Rcp) == int, "\nCP rank need to be int."

        As = [ np.random.rand(J[d], Rcp) for d in range(D) ]
        # Normalize factors s.t. the reconstraction is normalized
        for d in range(D):
            As[d] = 1.0 / (Rcp)**(1.0/D) * As[d] * 1.0 / As[d].sum(axis=0, keepdims=True)
        Qcp = CP.get_CP_Q(As)
        Pcp = CP.CP_from_factors(As)
        ranks["cp"] = Rcp

        # +1 because of the weight
        n_para_cp = utils.cp_n_params(J, Rcp) + 1
    else:
        Pcp = 0
        Ccp = 0
        eta_cp = 0
        As = 0
        n_para_cp = 0

    if not(learn_noise):
        eta_noise = 0

    ## Init for EMTucker
    if learn_tucker:
        if type(Rtucker) == int:
            print("\nRtucker need to be vector")
            Rtucker = [ Rtucker for d in range(D) ]
            print(f"Thus, Rtucker is modified as a vector {Rtucker}")
        else:
            assert len(Rtucker) == D, "Wrong dim of tucker rank"
        As_tucker = [ np.random.rand(J[d], Rtucker[d]) for d in range(D) ]
        G_tucker = np.random.rand(*Rtucker)
        Qtucker  = Tucker.get_Tucker_Q(G_tucker, As_tucker)
        Ptucker  = Tucker.Tucker_from_factors(G_tucker, As_tucker)
        indices_all_rnk = [ [rd for rd in range(Rtucker[d])] for d in range(D) ]
        ranks["tucker"] = Rtucker

        # +1 because of the weight
        n_para_tucker = utils.tucker_n_params(J, Rtucker)
    else:
        Ptucker = 0
        Ctucker = 0
        eta_tucker = 0
        As_tucker = 0
        G_tucker = 0
        n_para_tucker = 0

    n_params_total = n_para_cp + n_para_tucker + n_para_train
    n_params_dict  = {"cp":n_para_cp, "tucker":n_para_tucker, "train":n_para_train, "total":n_params_total}

    if learn_noise:
        n_params_total += 1

    P = eta_cp * Pcp + eta_tucker * Ptucker + eta_train * Ptrain + eta_noise / AbsOmegaI
    F = update_F(F,T,P,alpha)

    T_over_P = T**alpha * F / P
    sum_T_over_P = np.sum( T_over_P )

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
        N = np.count_nonzero(T)
        utils.print_verbose_top(alpha, N,
                n_para_cp, n_para_tucker, n_para_train, n_params_total,
                Rcp, Rtucker, R,
                learn_weights, learn_cp, learn_tucker, learn_train, learn_noise, sparse=False)

        """
        ## Show what low-rank structure will be used.
        print("\nEM mixture tensor learning for DENSE data") 
        objective = alpha
        if alpha == 1.0:
           objective = "KL"
        if alpha == 0.0:
            objective = "Reverse KL"

        print("Included low-rank structures:")
        if learn_cp:
            print(f"{'CPD':<10} {'n_params'}:{n_para_cp:<8} {'Rank':<5}:{Rcp:<5}")
        if learn_tucker:
            print(f"{'Tucker':<10} {'n_params'}:{n_para_tucker:<8} {'Rank':<5}:{Rtucker}")
        if learn_train:
            print(f"{'Train':<10} {'n_params'}:{n_para_train:<8} {'Rank':<5}:{R}")
        if learn_noise:
            print(f"{'Noise':<10}")

        print(f"{'Learn weights':<25}:{str(learn_weights):>10}.")

        nnz = np.count_nonzero(T)
        ## Show the number of parameters 
        print(f"\n{'Total number of params':<25}:{n_params_total:10d}")
        print(f"{'Number of non-zero values':<25}:{nnz:10d}")
        print(f"{'Objective function':<25}:{objective:>10}-div.")

        ## Header of verbose
        print(f"\n{'Iteration':<11} {'KL Error':<13} {'α-Error':<11} {'L2 Error':<14}  {'Weights':<8} {'CP':<6}  {'Tucker':<6}  {'Train':<7} {'Noise':<8}  {'Sum':<9}")
        """

    n_iter = 0

    ############################
    ############################
    ## Proposed EEM-algorithm ##
    ############################
    ############################

    while(n_iter < max_iter):
        #T_over_P = T / P

        ##################
        ## First E-STEP ##
        ##################
        F = update_F(F,T,P,alpha)

        for _ in range(iter_inside):

            #########################
            ## EM-algorithm inside ##
            #########################
            n_iter += 1

            T_over_P = T**alpha * F / P
            if learn_train:

                ########################
                ## UPDATE FOR EMTRAIN ##
                ########################

                GR = train.get_train_R(G) # G(  --> d )
                GL = train.get_train_L(G) # G( d <--  )

                ## Update G
                for d in range(D):
                    sum_axes = utils.tuple_skipping_m(D,d)
                    for rdm1, rd in product(range(r[d-1]), range(r[d])):
                        slice_GR = [slice(None)] * (GR[d-1].ndim - 1) + [rdm1] 
                        GR_new = np.tensordot(GR[d-1][tuple(slice_GR)], G[d][rdm1,:,rd], axes=0)

                        slice_GL = [rd] + [slice(None)] * (GL[d].ndim - 1)
                        X = np.tensordot(GR_new, GL[d][tuple(slice_GL)], axes=0)
                        G[d][rdm1,:,rd] = np.sum(T_over_P * X, axis=sum_axes ) / np.sum( T_over_P * X )

                ## Normalize G
                for d in range(D):
                    for rd in range(r[d]):
                        G[d][:,:,rd] /= np.sum( G[d][:,:,rd] )

                Ptrain  = train.train_from_cores_fast(G) 
                if update_rule == 0:
                    Ctrain  = eta_train * np.sum( T_over_P * Ptrain )
                else:
                    Ctrain  = np.sum( T_over_P * Ptrain )

            if learn_cp:

                ########################
                ## UPDATE FOR EMCP    ##
                ########################

                Qcp = CP.get_CP_Q(As)
                sum_rnk = [ np.sum(T_over_P * Qcp[r]) for r in range(Rcp) ]
                total = np.sum( sum_rnk )
                for d in range(D):
                    axis_to_sum = utils.tuple_skipping_m(D, d)
                    for rcp in range(Rcp):
                        As[d][:,rcp] = np.sum( Qcp[rcp] * T_over_P , axis=axis_to_sum) / ( total**(1/D) * sum_rnk[rcp] **(1-1/D) )
                        # You cannot leave total**(1/D) since it is no gurantee to be total == 1.

                Pcp     = CP.CP_from_factors(As)

                if update_rule == 0:
                    Ccp     = eta_cp * np.sum( T_over_P * Pcp )
                else:
                    Ccp     = np.sum( T_over_P * Pcp )



            if learn_tucker:

                #########################
                ## UPDATE FOR EMTucker ##
                #########################

                Qtucker = Tucker.get_Tucker_Q(G_tucker, As_tucker)

                # Update G_tucker
                for rtuc in product(*indices_all_rnk):
                    G_tucker[*rtuc] = np.sum( Qtucker[rtuc] * T_over_P )
                # normalize G_tucker
                G_tucker /= np.sum(G_tucker)

                # Update As_tucker
                for d in range(D):
                    axis_to_sum = utils.tuple_skipping_m(D, d)
                    for rd in range(Rtucker[d]):
                        indices_rnk = utils.get_rnk_indices_for_sum(d, rd, Rtucker)
                        As_tucker[d][:,rd] = sum( np.sum( Qtucker[rtucker] * T_over_P, axis=axis_to_sum ) for rtucker in product(*indices_rnk))

                    # Normalize As_tucker
                    for rd in range(Rtucker[d]):
                        As_tucker[d][:, rd] /= np.sum(As_tucker[d][:,rd])


                Ptucker = Tucker.Tucker_from_factors(G_tucker, As_tucker) 
                if update_rule == 0:
                    Ctucker = eta_tucker * np.sum( T_over_P * Ptucker )
                else:
                    Ctucker = np.sum( T_over_P * Ptucker )

            ## E-step

            if learn_noise:
                if update_rule == 1:
                    Cnoise = np.sum( T_over_P ) / AbsOmegaI
                else:
                    Cnoise = eta_noise * np.sum( T_over_P ) / AbsOmegaI
            else:
                Cnoise = 0.0

            if learn_weights:
                eta_cp     =     Ccp / (Ccp + Ctrain + Ctucker + Cnoise)
                eta_train  =  Ctrain / (Ccp + Ctrain + Ctucker + Cnoise)
                eta_tucker = Ctucker / (Ccp + Ctrain + Ctucker + Cnoise)
                eta_noise  =  Cnoise / (Ccp + Ctrain + Ctucker + Cnoise)

            P = eta_cp * Pcp + eta_train * Ptrain + eta_tucker * Ptucker + eta_noise / AbsOmegaI
            # To check if the normalization is satsified 
            # print(f" Total sum: {np.sum(P):.6f}" )
                
            if verbose:
                if n_iter % verbose_interval == 0:
                    kl_error = utils.KL_div(T,P,avoid_nan=avoid_nan)
                    alpha_error = utils.alpha_div(T,P,alpha,avoid_nan=avoid_nan)
                    #if np.isnan(kl_error):
                    #    print(f"objective function becomes NaN, values:{kl_error}")

                    nl_error = utils.NL(T,P,avoid_nan=avoid_nan)
                    f_error  = np.linalg.norm(T-P)/np.linalg.norm(T)
                    #print(n_iter, noise, f_error, kl_error)
                    #print(n_iter, kl_error, eta_cp, eta_tucker, eta_train, eta_noise)
                    if np.isnan(f_error):
                        print(f"L2 norm becomes NaN, values:{f_error}")
                        break
                        
                    elapsed_time = time.perf_counter() - start_time
                    print(f"Iter: {n_iter:5d} KL: {kl_error:.7f} α:{alpha_error:.7f} L2: {f_error:.7f} | Weights: {eta_cp:.5f} {eta_tucker:.5f} {eta_train:.5f} {eta_noise:.5f} | {np.sum(P):.2f} | {elapsed_time:.2f} sec.")
                    if prev_error < alpha_error:
                        print("alpha-div. is not monotonicaly decreasing")
                    prev_error = alpha_error

                    if loss_history:
                        loss_kl_history.append(kl_error)
                        loss_nl_history.append(nl_error)
                        loss_alpha_history.append(alpha_error)
                        loss_fro_history.append(f_error)
                        iter_history.append(n_iter)
                        elapsed_times.append(elapsed_time)

            if n_iter > 3 and n_iter % conv_check_interval == 1 and n_iter > verbose_interval:
                kl_error = utils.KL_div(T,P,avoid_nan=avoid_nan)
                res = abs( prev_error_for_conv - alpha_error ) / conv_check_interval
                if res < tol:
                    break
                else:
                    prev_error_for_conv = alpha_error



    hisotry = {"iter":iter_history, "kl":loss_kl_history, "alpha":loss_alpha_history, "nl":loss_nl_history, "fro":loss_fro_history, "time":elapsed_times}
    factors = {"As":As, "G_tucker":G_tucker, "As_tucker":As_tucker, "G_train":G, "eta_cp":eta_cp, "eta_tucker":eta_tucker, "eta_train":eta_train, "eta_noise":eta_noise}

    models  = [learn_cp, learn_tucker, learn_train, learn_noise]
    details = {"max_iter":max_iter, "n_iter":n_iter, "tol":tol, "verbose_interval":verbose_interval, "n_params":n_params_dict,
               "conv_check_interval":conv_check_interval, "models":models, "rank":ranks, "update_rule":update_rule}

    # P is the reconst (mixture) low-rank tensor
    # If you wanna reconst tensor from factors, run
    # utils_mix.reconst_all(factors)
    return factors, P, hisotry, details

def update_F(F,T,P,alpha):
    F = P**(1-alpha) / np.sum( (T**alpha) * (P**(1-alpha)) )
    return F

