## NOTE:
## As in sparse and in dnese have 
## different data structure
## Thus, recons_all can be applicatable only for dense

def reconst_all(factors, model):
    learn_cp, learn_tucker, learn_train, learn_noise = model

    As = factors["As"]
    if learn_cp and As != 0:
        eta_cp = factors["eta_cp"]
        Pcp = eta_cp * CP.CP_from_factors(As)
        tensor_dim  = len(As)
        tensor_size = [np.shape(As[d])[0] for d in range(tensor_dim)]
        print(tensor_dim, tensor_size)
    else:
        print("No CP structure in this mixture")
        Pcp = 0
        
    G_tucker = factors["G_tucker"]
    if learn_tucker and type(G_tucker) != int:
        eta_tucker = factors["eta_tucker"]
        As_tucker = factors["As_tucker"]
        Ptucker = eta_tucker* Tucker.Tucker_from_factors(G_tucker, As_tucker)
    
        tensor_dim = len(As_tucker)
        tensor_size = [np.shape(As_tucker[d])[0] for d in range(tensor_dim)]
        print(tensor_dim, tensor_size)
    else:
        print("No Tucker structure in this mixture")
        Ptucker = 0
        
    G_train = factors["G_train"]
    if learn_train and type(G_train) != int:
        eta_train = factors["eta_train"]
        Ptrain  = eta_train * train.train_from_cores(G_train)
        
        tensor_dim = len(G_train)
        tensor_size = [ np.shape(G_train[d])[1] for d in range(tensor_dim) ]
        print(tensor_dim, tensor_size)
    else:
        print("No Train structure in this mixture")
        Ptrain = 0

    if learn_noise:
        eta_noise = factors["eta_noise"]
        AbsOmegaI = np.prod(tensor_size)
        Pnoise = eta_noise / AbsOmegaI
    else:
        print("No noise parameter in this mixture")
        Pnoise = 0
    
    P = Pcp + Ptucker + Ptrain + Pnoise
    return P