import torch
from python_code.image_manipulation import *
import os
from python_code.utils import get_weights_from_snaphu
from python_code.parameters import ModelParameters, IrlsParameters



DEFAULT_TYPE = torch.float32



def unwrap(
    X,
    model_params: ModelParameters=ModelParameters(),
    irls_params: IrlsParameters=IrlsParameters(),
    amp1=None, amp2=None, corrfile=None,
    Cv=None, Ch=None, 
    weighting_strategy="snaphu_weights",
    snaphu_config_file=None, snaphu_bin=None,
    run_on_gpu: bool=True,
    verbose=True):
    """Unwraps an image X.

    Arguments:
        X: 2darray (N, M)
            image to be unwrapped
        model_params: ModelParameters, optional
            parameters of the model (default is ModelParameters())
        irls_params: IrlsParams, optional
            parameters for running the IRLS algorithm (default is IrlsParameters())
        amp1: 2darray (N, M), optional
            amplitude of the first image in the interferogram. Is used to compute weights if available (default is None)
        amp2: 2darray (N, M), optional
            amplitude of the second image in the interferogram. Is used to compute weights if available (default is None)
        corrfile: 2darray (N, M), optional
            coherence map of the interferogram. Is used to compute weights if available (default is None)
        Cv: 2darray (N-1, M), optional.
            user-supplied weights corresponding to vertical wrapped gradients (default is None).
        Ch: 2darray (N, M-1), optional.
            user-supplied weights corresponding to horizontal wrapped gradients (default is None).
        weighting_strategy: str, optional.
            If Cv and Ch are None, strategy for computing them. Should either be "uniform", "snaphu_weights" or None (default is "snaphu_weights").
        snaphu_config_file: str, optional.
            Path to a SNAPHU config file to compute the weights when weighting_strategy is et to "snaphu_weights".
            See SNAPHU documentation and "https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu.conf.full" for more details on this configuration file.
            (Default is None).
        snaphu_bin: str, optional.
            path to SNAPHU executable.
        verbose: Bool, optional (default is True).        

    Returns:
        U: 2darray (N, M)
            unwrapped image
    """

    N, M = X.shape
    print(N, M)

    if Cv is not None and Ch is not None:
        if Cv.shape == (N-1, M) and Ch.shape == (N, M-1):
            print("Using user-supplied weights.")
        else:
            raise RuntimeError("Supplied weights do not have appropriate shape.")
    elif weighting_strategy is None or weighting_strategy=="uniform":
        Cv = np.ones((N-1, M))
        Ch = np.ones((N, M-1))
        print("Using uniform unity weights.")
    elif weighting_strategy=="snaphu_weights":
        if snaphu_bin is None:
            raise RuntimeError("Cannot compute weights with SNAPHU if SNAPHU path is not specified.")
        elif not os.path.isfile(snaphu_bin):
            raise RuntimeError(f"Cannot find snaphu bin at specified location {snaphu_bin}")
        else:
            print("Computing statistical-based weights using SNAPHU")
            tmpdir = os.path.join(os.getcwd(), "snaphu_weights_tmp_dir") #will be removed anyway
            Cv, Ch = get_weights_from_snaphu(X.astype(np.float32), tmpdir, amp1=amp1, amp2=amp2, corrfile=corrfile, snaphu_bin=snaphu_bin, snaphu_config_file=snaphu_config_file)
            Ch[0, 0] = np.median(Cv)
            Ch[0, -1] = np.median(Cv)
            Ch[-1, 0] = np.median(Cv)
            Ch[-1, -1] = np.median(Cv)

            Cv = Cv.astype(float)
            Ch = Ch.astype(float)
            max_value = np.max([np.max(Cv), np.max(Ch)])
            Cv /= float(max_value)
            Ch /= float(max_value)
    else:
        raise NotImplementedError("Weighting strategy {} unknwon".format(weighting_strategy))


    U, Vv, Vh = IRLS(X, Cv, Ch, model_params=model_params, irls_params=irls_params, run_on_gpu=run_on_gpu, verbose=verbose)

    return U, Vv, Vh



def IRLS(X, Cv, Ch, model_params: ModelParameters=ModelParameters(),
        irls_params: IrlsParameters=IrlsParameters(), 
        run_on_gpu=True,
        verbose=True):
    """Performs L1-norm minimization phase unwrapping using IRLS algorithm on image X with weights Ch and Cv.

    Arguments:
        X: 2darray (N, M)
            image to be unwrapped
        model_params: ModelParameters, optional
            parameters of the model (default is ModelParameters())
        irls_params: IrlsParams, optional
            parameters for running the IRLS algorithm (default is IrlsParameters())
        Cv: 2darray (N-1, M)
            weights cooresponding to vertical gradients
        Ch: 2darray (N, M-1)
            weights cooresponding to horizontal gradients
        verbose: Bool, optional (default is True).        

    Returns:
        U: 2darray (N, M)
            unwrapped image
        Vv: 2darray (N-1, M)
            vertical penalty term
        Vh: 2darray (N, M-1)
            horizontal pernalty term
    """
    
    N, M = X.shape

    tau = model_params.tau
    delta = model_params.delta

    max_iter = irls_params.max_iter
    preconditioner_name = irls_params.preconditioner
    max_iter_CG_strategy = irls_params.max_iter_CG_strategy
    max_iter_CG = irls_params.max_iter_CG
    rel_tol_CG = irls_params.rel_tol_CG
    abs_tol_CG = irls_params.abs_tol_CG


    if max_iter_CG_strategy != "heuristics" and max_iter_CG_strategy != "constant":
        raise NotImplementedError("Strategy {} for updating max number of CG iterations unknown.".format(max_iter_CG_strategy))

    S = build_S(N)
    T = build_T(M)
    Gv = wrap_matrix(apply_S(X))
    Gh = wrap_matrix(apply_T(X))

    Gv = torch.tensor(Gv, dtype=DEFAULT_TYPE)
    Gh = torch.tensor(Gh, dtype=DEFAULT_TYPE)
    S = torch.tensor(S, dtype=DEFAULT_TYPE)
    T = torch.tensor(T, dtype=DEFAULT_TYPE)

    Cv = torch.tensor(Cv, dtype=DEFAULT_TYPE)
    Ch = torch.tensor(Ch, dtype=DEFAULT_TYPE)

    Wv = torch.ones((N-1, M), dtype=DEFAULT_TYPE)
    Wh = torch.ones((N, M-1), dtype=DEFAULT_TYPE)
    U = torch.zeros((N, M), dtype=DEFAULT_TYPE)


    if run_on_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print("Running on CUDA")
    elif run_on_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Running on Metal (MPS)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("WARNING: no GPU device found. Running on CPU. This will significantly impact performance")

    S = S.to(device)
    T = T.to(device)
    U = U.to(device)
    Gv = Gv.to(device)
    Gh = Gh.to(device)
    Wv = Wv.to(device)
    Wh = Wh.to(device)
    Cv = Cv.to(device)
    Ch = Ch.to(device)


    S = S.to_sparse()
    T = T.to_sparse()

    Vv = S @ U - Gv
    Vh = U @ T - Gh

    preconditioner_constant_part = None

    if preconditioner_name == "block_diag":
        DS, PS = torch.linalg.eigh((S.T @ S).to_dense(), UPLO="L")
        DS = torch.where(DS > 1e-4, DS, 1e-4)

        if N == M:
            PT = PS
            DT = DS
        else:
            DT, PT = torch.linalg.eigh((T @ T.T).to_dense(), UPLO="L")
            DT = torch.where(DT > 1e-4, DT, 1e-4)

        if verbose:
            print("Done computing decomposition of S'S and of TT'...")

        kron_diag = get_diagonal_kron_identity(DS.cpu().numpy(), "left", M) + get_diagonal_kron_identity(DT.cpu().numpy(), "right", N)
        #kron_diag = torch.kron(torch.eye(M), torch.diag(DS)) + torch.kron(torch.diag(DT), torch.eye(N))

        kron_diag = kron_diag.reshape((N, M), order='F')
        kron_diag = torch.tensor(kron_diag, dtype=DEFAULT_TYPE)

        PS_transpose = PS.T
        PT_transpose = PT.T
 
        DS = DS.to(device)
        DT = DT.to(device)
        PS = PS.to(device)
        PT = PT.to(device)
        PS_transpose = PS_transpose.to(device)
        PT_transpose = PT_transpose.to(device)
        kron_diag = kron_diag.to(device)

        preconditioner_constant_part = (PS, PS_transpose, PT, PT_transpose, kron_diag)




    tau = torch.tensor(tau, dtype=DEFAULT_TYPE)
    delta = torch.tensor(delta, dtype=DEFAULT_TYPE)
    
    #if torch.cuda.is_available():
    #    tau = tau.cuda()
    #    delta = delta.cuda()
    tau = tau.to(device)
    delta = delta.to(device)

    #Wh = 1.0 / torch.sqrt(Ch**2 * Vh**2 + delta**2)
    #Wv = 1.0 / torch.sqrt(Cv**2 * Vv**2 + delta**2)
    Wv =  torch.sqrt(Cv**2 * Vv**2 + delta**2)
    Wh =  torch.sqrt(Ch**2 * Vh**2 + delta**2)

    F_delta_prev = F_delta(delta=delta, Vv=Vv, Vh=Vh, U=U, Wv=Wv, Wh=Wh,
                                       tau=tau, Gv=Gv, Gh=Gh, S=S, T=T, Cv=Cv, Ch=Ch)
    if verbose:
        print("F_delta at start =", F_delta_prev.item())
    

    just_increased_max_iter_CG = False
    if max_iter_CG_strategy == "heuristics":
        next_max_iter_CG = irls_params.max_iter_CG_start
    elif max_iter_CG_strategy == "constant":
        next_max_iter_CG = max_iter_CG
    stepsize = delta

    for iteration in range(1, int(max_iter) + 1):
        if verbose:
            print("########################### Iteration", iteration, ", delta =", delta.item(), "#####################################")
            print(torch.sum(U).item())

        # Update Vv, Vh, U with CG
        Vv_prev = Vv
        Vh_prev = Vh
        U_prev = U

        if verbose:
            print("Maximum number of CG iterations = ", next_max_iter_CG)
        Vv, Vh, U = CG(Cv**2 * ( 1.0  / Wv), Ch**2 * ( 1.0 / Wh), S, T,
                                        tau, Gv, Gh,
                                        Vv, Vh, U,
                                        max_iter_CG=next_max_iter_CG,
                                        preconditioner_name=preconditioner_name,
                                        preconditioner_constant_part=preconditioner_constant_part,
                                        abs_tol=abs_tol_CG, rel_tol=rel_tol_CG,
                                        verbose=verbose)

        if verbose:
            print("Sum(U) = ", torch.sum(U).item())
        U = U - torch.mean(U)
        F_delta_new = F_delta(delta=delta, Vv=Vv, Vh=Vh, U=U, Wv=Wv, Wh=Wh, tau=tau, Gv=Gv, Gh=Gh, S=S, T=T, Cv=Cv, Ch=Ch)
        relative_improvement = (F_delta_prev - F_delta_new) / F_delta_prev
        if verbose:
            print("F_delta after updating Vh, Vv, U =", F_delta_new.item(), ", relative improvement =", relative_improvement.item())

        F_delta_prev = F_delta_new

        # Update weights
        #Wh = 1.0 / torch.sqrt(Ch**2 * Vh**2 + delta**2)
        #Wv = 1.0 / torch.sqrt(Cv**2 * Vv**2 + delta**2)
        Wv =  torch.sqrt(Cv**2 * Vv**2 + delta**2)
        Wh =  torch.sqrt(Ch**2 * Vh**2 + delta**2)

        F_delta_new = F_delta(delta=delta, Vv=Vv, Vh=Vh, U=U, Wv=Wv, Wh=Wh, tau=tau, Gv=Gv, Gh=Gh, S=S, T=T, Cv=Cv, Ch=Ch)
        relative_improvement = (F_delta_prev - F_delta_new) / F_delta_prev
        if verbose:
            print("F_delta after updating Wv, Wh =", F_delta_new.item(), ", relative_improvement =", relative_improvement.item())
            print("min(Wv) = ", torch.min(Wv).item(), ", max(Wh) = ", torch.max(Wh).item())

        F_delta_prev = F_delta_new
        if max_iter_CG_strategy == "heuristics":
            if relative_improvement < irls_params.rel_improvement_tol:
                if just_increased_max_iter_CG:
                    break
                else:
                    next_max_iter_CG = min(int(np.round(next_max_iter_CG * irls_params.increase_CG_max_iteration_factor)), max_iter_CG)
                    just_increased_max_iter_CG = True
            else:
                just_increased_max_iter_CG = False

    return U.cpu().numpy(), Vv.cpu().numpy(), Vh.cpu().numpy()


def CG(Qv, Qh, S, T, tau, Gv, Gh, Vv_start, Vh_start, U_start,
                        max_iter_CG, preconditioner_name, preconditioner_constant_part,
                        abs_tol=1e-10, rel_tol=0, verbose=True):

    """Performs conjugate gradient method to solve linear system appearing in the IRLS iterations.

    Arguments:
        Qv: 2darray (N-1, M)
            constant matrix involved in the hadamard product of the vertical part of the linear map, i.e. should be Qv = Cv * Cv * (1/Wv).
        Qv: 2darray (N, M-1)
            constant matrix involved in the hadamard product of the horizontal part of the linear map, i.e. should be Qh = Ch * Ch * (1/Wh).
        S: 2darray(N-1, N)
            vertical shift array
        T: 2darray(M, M-1)
            horizontal shift array
        tau: float
            quadratic penalization term
        Gv: 2darray (N-1, M)
            vertical wrapped gradients
        Gh: 2darray (N, M-1)
            horizontal wrapped gradients
        Vv_start: 2darray (N-1, M)
            initial value for vertical Vv term
        Vh_start: 2darray (N, M-1)
            initial value for horizontal Vh term
        U_start: 2darray (N, M)
            initial value for U term
        max_iter_CG: int
            maximum number of iterations
        preconditioner_name: str
            preconditioner name. Should be either None or "block_diag"
        preconditioner_constant_part: tuple
            tuple containing useful precomputed factorizations for applying preconditioner
        abs_tol: float
            stop algorithm when residue norm is smaller than this value (default is 1e-10)
        rel_tol: float:
            stop algorithm when relative residue norm is smaller than this value (default is 0)
        verbose: Bool, optional (default is True).        

    Returns:
        Vh: 2darray (N-1, M)
            horizontal Vh term
        Vv: 2darray (N, M-1)
            vertical Vv term
        U: 2darray (N, M)
            U term
    """

    N = S.shape[0]
    M = T.shape[1]

    Vv = Vv_start
    Vh = Vh_start
    U = U_start

    preconditioner_variable_part = get_preconditioner(preconditioner_name=preconditioner_name,
                                                                    S=S, T=T, Qv=Qv, Qh=Qh, tau=tau)

    B_v = -1 / tau * Gv
    B_h = -1 / tau * Gh
    B_U = 1 / tau * (S.T @ Gv + Gh @ T.T)

    linear_map_0_v, linear_map_0_h, linear_map_0_U = apply_linear_map(Vv_start, Vh_start, U_start, Qv, Qh, tau, S, T)
    R_k_v = B_v - linear_map_0_v
    R_k_h = B_h - linear_map_0_h
    R_k_U = B_U - linear_map_0_U

    Z_k_v, Z_k_h, Z_k_U = apply_preconditioner(R_k_v, R_k_h, R_k_U, tau=tau,
                                                            preconditioner_name=preconditioner_name,
                                                            preconditioner_constant_part=preconditioner_constant_part,
                                                            preconditioner_variable_part=preconditioner_variable_part)


    P_k_v = Z_k_v
    P_k_h = Z_k_h
    P_k_U = Z_k_U

    rho_k = torch.sum(R_k_v * Z_k_v) + torch.sum(R_k_h * Z_k_h) + torch.sum(R_k_U * Z_k_U)

    norm_r_0 = torch.sqrt(torch.sum(R_k_v**2) + torch.sum(R_k_h**2) + torch.sum(R_k_U**2))

    for iteration_CG in range(1, int(max_iter_CG) + 1):

        linear_map_k_v, linear_map_k_h, linear_map_k_U = apply_linear_map(P_k_v, P_k_h, P_k_U, Qv, Qh, tau, S, T)

        out = torch.trace(P_k_v.T @ linear_map_k_v) + torch.trace(P_k_h.T @ linear_map_k_h) + torch.trace(P_k_U.T @ linear_map_k_U)
        alpha_k = rho_k / out

        Vv = Vv + (alpha_k * P_k_v)
        Vh = Vh + (alpha_k * P_k_h)
        U = U + (alpha_k * P_k_U)

        R_k1_v = R_k_v - (alpha_k * linear_map_k_v)
        R_k1_h = R_k_h - (alpha_k * linear_map_k_h)
        R_k1_U = R_k_U - (alpha_k * linear_map_k_U)



        Z_k1_v, Z_k1_h, Z_k1_U = apply_preconditioner(R_k1_v, R_k1_h, R_k1_U, tau=tau,
                                                                    preconditioner_name=preconditioner_name,
                                                                    preconditioner_constant_part=preconditioner_constant_part,
                                                                    preconditioner_variable_part=preconditioner_variable_part)

        rho_k1 = torch.sum(R_k1_v * Z_k1_v) + torch.sum(R_k1_h * Z_k1_h) + torch.sum(R_k1_U * Z_k1_U)
        beta = rho_k1 / rho_k

        P_k1_v = Z_k1_v + (beta * P_k_v)
        P_k1_h = Z_k1_h + (beta * P_k_h)
        P_k1_U = Z_k1_U + (beta * P_k_U)

        # updates
        R_k_v = R_k1_v
        R_k_h = R_k1_h
        R_k_U = R_k1_U

        P_k_v = P_k1_v
        P_k_h = P_k1_h
        P_k_U = P_k1_U

        rho_k = rho_k1

        Z_k_v = Z_k1_v
        Z_k_h = Z_k1_h
        Z_k_U = Z_k1_U

        residue_norm = torch.sqrt(torch.sum(R_k_v**2) + torch.sum(R_k_h**2) + torch.sum(R_k_U**2))
        relative_residue_norm = residue_norm / norm_r_0

        if iteration_CG % (max_iter_CG / 10) == 0:
            if verbose:
                print("    CG iteration", iteration_CG, "residue norm =", residue_norm.item(), "relative residue norm =", relative_residue_norm.item())

        if residue_norm < abs_tol or relative_residue_norm < rel_tol:
            if verbose:
                print("    Breaking at CG iteration", iteration_CG, "residue norm =", residue_norm.item(), "relative residue norm =", relative_residue_norm.item())
            break

    return Vv, Vh, U



def apply_linear_map(Vv, Vh, U, Qv, Qh, tau, S, T):
    SU_minus_Vv = 1/tau * (S @ U - Vv)
    UT_minus_Vh = 1/tau * (U @ T - Vh)
    top_part = Qv * Vv - SU_minus_Vv
    middle_part = Qh * Vh - UT_minus_Vh
    bottom_part = S.T @ SU_minus_Vv
    bottom_part = bottom_part + UT_minus_Vh @ T.T
    return top_part, middle_part, bottom_part  




def get_diagonal_kron_identity(K, side, identity_size):
    """computes the diagonal of kron(np.eye(identity_size), np.diag(K)) if side is "left" and diagonal of kron(np.diag(K), np.eye(identity_size)) if side is "right"

    Arguments:
        K: 1darray (N)
            1d-vector representing diagonal matrix of which we want to compute the kronecker product
        side: str
            string detailing on which side the identity should be in the kronecker product. Should be either "left" or "right"
        identity_size: int
            size of the identity matrix in the kronecker product
    
    Returns:
        res: 1darray (N*identity_size)
            diagonal of the corresponding kronecker product
    """

    # K is a one-dimensional array
    # returns the diagonal of
    # kron(np.eye(identity_size), np.diag(K)) if side = "left"
    # kron(np.diag(K), np.eye(identity_size)) if side = "right"
    K_size = K.shape[0]
    res = np.zeros(K_size*identity_size)
    if side == "left":
        for i in range(0, K_size*identity_size, K_size):
            res[i:i+K_size] = K
    elif side == "right":
        for i in range(K_size):
            res[i*identity_size : (i+1)*identity_size] = K[i]

    return res

def get_preconditioner(preconditioner_name, S, T, Qv, Qh, tau):
    if preconditioner_name == "None":
        return None
    elif preconditioner_name == "block_diag":
        Vv_part = Qv + 1/tau 
        Vh_part = Qh + 1/tau 

        return (Vv_part, Vh_part)


def apply_preconditioner(R_v, R_h, R_U, tau, preconditioner_variable_part,
                                      preconditioner_constant_part, preconditioner_name):
    if preconditioner_name == "None":
        return R_v, R_h, R_U
    elif preconditioner_name == "block_diag":
        Vv_part, Vh_part = preconditioner_variable_part

        PS, PS_transpose, PT, PT_transpose, kron_diag = preconditioner_constant_part

        res_v = R_v / Vv_part
        res_h = R_h / Vh_part

        res_U = solve_sylvester(R_U, PS, PS_transpose, PT, PT_transpose, kron_diag, tau)
        return res_v, res_h, res_U


def solve_sylvester(R, PS, PS_transpose, PT, PT_transpose, kron_diag, tau):
    RHS = PS_transpose @ (tau * R) @ PT
    PUP = RHS / kron_diag

    Res = PS @ PUP @ PT_transpose
    return Res



def F_delta(delta, Vv, Vh, U, Wv, Wh, tau, Gv, Gh, S, T, Cv=0, Ch=0):
    term_v = 0.5 * torch.sum( ((Cv * Vv)**2 + delta**2) * (1.0 / Wv) +  Wv)
    term_h = 0.5 * torch.sum( ((Ch * Vh)**2 + delta**2) * (1.0 / Wh) +  Wh)
    term_reg = 0.5 / tau * ( torch.sum( (Vv - S@U + Gv)**2) + torch.sum( (Vh - U@T + Gh)**2))
    return term_v + term_h + term_reg