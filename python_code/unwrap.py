import torch
from python_code.image_manipulation import *
#import torch_dct as dct
import os
from python_code.utils import get_weights_from_snaphu
from python_code.parameters import ModelParameters, IrlsParameters



DEFAULT_TYPE = torch.float32



def unwrap(
    X,
    model_params: ModelParameters=ModelParameters(),
    irls_params: IrlsParameters=IrlsParameters(),
    amp1=None, amp2=None, corrfile=None,
    Ch=None, Cv=None, 
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
        Ch: 2darray (N-1, M), optional.
            user-supplied weights (default is None).
        Cv: 2darray (N, M-1), optional.
            user-supplied weights (default is None).
        weighting_strategy: str, optional.
            If Ch and Cv are None, strategy for computing them. Should either be "uniform", "snaphu_weights" or None (default is "snaphu_weights").
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

    if Ch is not None and Cv is not None:
        if Ch.shape == (N-1, M) and Cv.shape == (N, M-1):
            print("Using user-supplied weights.")
        else:
            raise RuntimeError("Supplied weights do not have appropriate shape.")
    elif weighting_strategy is None or weighting_strategy=="uniform":
        Ch = np.ones((N-1, M))
        Cv = np.ones((N, M-1))
        print("Using uniform unity weights.")
    elif weighting_strategy=="snaphu_weights":
        if snaphu_bin is None:
            raise RuntimeError("Cannot compute weights with SNAPHU if SNAPHU path is not specified.")
        elif not os.path.isfile(snaphu_bin):
            raise RuntimeError(f"Cannot find snaphu bin at specified location {snaphu_bin}")
        else:
            print("Computing statistical-based weights using SNAPHU")
            tmpdir = os.path.join(os.getcwd(), "snaphu_weights_tmp_dir") #will be removed anyway
            Ch, Cv = get_weights_from_snaphu(X.astype(np.float32), tmpdir, amp1=amp1, amp2=amp2, corrfile=corrfile, snaphu_bin=snaphu_bin, snaphu_config_file=snaphu_config_file)
            Cv[0, 0] = np.median(Cv)
            Cv[0, -1] = np.median(Cv)
            Cv[-1, 0] = np.median(Cv)
            Cv[-1, -1] = np.median(Cv)

            Ch = Ch.astype(float)
            Cv = Cv.astype(float)
            max_value = np.max([np.max(Ch), np.max(Cv)])
            Ch /= float(max_value)
            Cv /= float(max_value)
    else:
        raise NotImplementedError("Weighting strategy {} unknwon".format(weighting_strategy))


    U, Vh, Vv = IRLS(X, Ch, Cv, model_params=model_params, irls_params=irls_params, run_on_gpu=run_on_gpu, verbose=verbose)

    return U, Vh, Vv



def IRLS(X, Ch, Cv, model_params: ModelParameters=ModelParameters(),
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
        Ch: 2darray (N-1, M)
            weights cooresponding to horizontal gradients
        Cv: 2darray (N, M-1)
            weights cooresponding to vertical gradients
        verbose: Bool, optional (default is True).        

    Returns:
        U: 2darray (N, M)
            unwrapped image
        Vh: 2darray (N-1, M)
            horizontal penalty term
        Vv: 2darray (N, M-1)
            vertical pernalty term
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
    Gh = wrap_matrix(apply_S(X))
    Gv = wrap_matrix(apply_T(X))

    Gh = torch.tensor(Gh, dtype=DEFAULT_TYPE)
    Gv = torch.tensor(Gv, dtype=DEFAULT_TYPE)
    S = torch.tensor(S, dtype=DEFAULT_TYPE)
    T = torch.tensor(T, dtype=DEFAULT_TYPE)

    Ch = torch.tensor(Ch, dtype=DEFAULT_TYPE)
    Cv = torch.tensor(Cv, dtype=DEFAULT_TYPE)

    Wh = torch.ones((N-1, M), dtype=DEFAULT_TYPE)
    Wv = torch.ones((N, M-1), dtype=DEFAULT_TYPE)
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
    Gh = Gh.to(device)
    Gv = Gv.to(device)
    Wh = Wh.to(device)
    Wv = Wv.to(device)
    Ch = Ch.to(device)
    Cv = Cv.to(device)


    S = S.to_sparse()
    T = T.to_sparse()

    Vh = S @ U - Gh
    Vv = U @ T - Gv

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

    elif preconditioner_name == "block_diag_dct":
        DS = torch.diag(dct.dct_2d(S.T @ S))
        DS = torch.where(DS > 1e-6, DS, 1e-6)

        if N == M:
            DT = DS
        else:
            DT, PT = torch.linalg.eigh((T @ T.T).to_dense(), UPLO="L")
            #DT = torch.where(TT_eigenvalues > 1e-6, TT_eigenvalues, 1e-6)
        if verbose:
            print("Done computing DS and DT")

        kron_diag = get_diagonal_kron_identity(DS.cpu().numpy(), "left", M) + get_diagonal_kron_identity(DT.cpu().numpy(), "right", N)
        #kron_diag = torch.kron(torch.eye(M), torch.diag(DS)) + torch.kron(torch.diag(DT), torch.eye(N))

        kron_diag = kron_diag.reshape((N, M), order='F')
        kron_diag = torch.tensor(kron_diag, dtype=DEFAULT_TYPE)
        #if torch.cuda.is_available():
        #    kron_diag = kron_diag.cuda()
        kron_diag = kron_diag.to(device)
        preconditioner_constant_part = kron_diag




    tau = torch.tensor(tau, dtype=DEFAULT_TYPE)
    delta = torch.tensor(delta, dtype=DEFAULT_TYPE)
    
    #if torch.cuda.is_available():
    #    tau = tau.cuda()
    #    delta = delta.cuda()
    tau = tau.to(device)
    delta = delta.to(device)

    #Wh = 1.0 / torch.sqrt(Ch**2 * Vh**2 + delta**2)
    #Wv = 1.0 / torch.sqrt(Cv**2 * Vv**2 + delta**2)
    Wh =  torch.sqrt(Ch**2 * Vh**2 + delta**2)
    Wv =  torch.sqrt(Cv**2 * Vv**2 + delta**2)

    F_delta_prev = F_delta(delta=delta, Vh=Vh, Vv=Vv, U=U, Wh=Wh, Wv=Wv,
                                       tau=tau, Gh=Gh, Gv=Gv, S=S, T=T, Ch=Ch, Cv=Cv)
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

        # Update Vh, Vv, Vpsi with CG
        Vh_prev = Vh
        Vv_prev = Vv
        U_prev = U

        if verbose:
            print("Maximum number of CG iterations = ", next_max_iter_CG)
        Vh, Vv, U = CG(Ch**2 * ( 1.0  / Wh), Cv**2 * ( 1.0 / Wv), S, T,
                                        tau, Gh, Gv,
                                        Vh, Vv, U,
                                        max_iter_CG=next_max_iter_CG,
                                        preconditioner_name=preconditioner_name,
                                        preconditioner_constant_part=preconditioner_constant_part,
                                        abs_tol=abs_tol_CG, rel_tol=rel_tol_CG,
                                        verbose=verbose)

        if verbose:
            print("Sum(U) = ", torch.sum(U).item())
        U = U - torch.mean(U)
        F_delta_new = F_delta(delta=delta, Vh=Vh, Vv=Vv, U=U, Wh=Wh, Wv=Wv, tau=tau, Gh=Gh, Gv=Gv, S=S, T=T, Ch=Ch, Cv=Cv)
        relative_improvement = (F_delta_prev - F_delta_new) / F_delta_prev
        if verbose:
            print("F_delta after updating Vh, Vv, U =", F_delta_new.item(), ", relative improvement =", relative_improvement.item())

        F_delta_prev = F_delta_new

        # Update weights
        #Wh = 1.0 / torch.sqrt(Ch**2 * Vh**2 + delta**2)
        #Wv = 1.0 / torch.sqrt(Cv**2 * Vv**2 + delta**2)
        Wh =  torch.sqrt(Ch**2 * Vh**2 + delta**2)
        Wv =  torch.sqrt(Cv**2 * Vv**2 + delta**2)

        F_delta_new = F_delta(delta=delta, Vh=Vh, Vv=Vv, U=U, Wh=Wh, Wv=Wv, tau=tau, Gh=Gh, Gv=Gv, S=S, T=T, Ch=Ch, Cv=Cv)
        relative_improvement = (F_delta_prev - F_delta_new) / F_delta_prev
        if verbose:
            print("F_delta after updating Wh, Wv =", F_delta_new.item(), ", relative_improvement =", relative_improvement.item())
            print("min(Wh) = ", torch.min(Wh).item(), ", max(Wh) = ", torch.max(Wh).item())

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

    return U.cpu().numpy(), Vh.cpu().numpy(), Vv.cpu().numpy()


def CG(Qh, Qv, S, T, tau, Gh, Gv, Vh_start, Vv_start, U_start,
                        max_iter_CG, preconditioner_name, preconditioner_constant_part,
                        abs_tol=1e-10, rel_tol=0, verbose=True):

    """Performs conjugate gradient method to solve linear system appearing in the IRLS iterations.

    Arguments:
        Qh: 2darray (N-1, M)
            constant matrix involved in the hadamard product of the horizontal part of the linear map, i.e. should be Qh = Ch * Ch * (1/Wh).
        Qv: 2darray (N, M-1)
            constant matrix involved in the hadamard product of the vertical part of the linear map, i.e. should be Qv = Cv * Cv * (1/Wv).
        S: 2darray(N-1, N)
            horizontal shift array
        T: 2darray(M, M-1)
            vertical shift array
        tau: float
            quadratic penalization term
        Gh: 2darray (N-1, M)
            horizontal wrapped gradients
        Gv: 2darray (N, M-1)
            vertical wrapped gradients
        Vh_start: 2darray (N-1, M)
            initial value for horizontal Vh term
        Vv_start: 2darray (N, M-1)
            initial value for vertical Vv term
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

    Vh = Vh_start
    Vv = Vv_start
    U = U_start

    preconditioner_variable_part = get_preconditioner(preconditioner_name=preconditioner_name,
                                                                    S=S, T=T, Qh=Qh, Qv=Qv, tau=tau)

    B_h = -1 / tau * Gh
    B_v = -1 / tau * Gv
    B_U = 1 / tau * (S.T @ Gh + Gv @ T.T)

    linear_map_0_h, linear_map_0_v, linear_map_0_U = apply_linear_map(Vh_start, Vv_start, U_start, Qh, Qv, tau, S, T)
    R_k_h = B_h - linear_map_0_h
    R_k_v = B_v - linear_map_0_v
    R_k_U = B_U - linear_map_0_U

    Z_k_h, Z_k_v, Z_k_U = apply_preconditioner(R_k_h, R_k_v, R_k_U, tau=tau,
                                                            preconditioner_name=preconditioner_name,
                                                            preconditioner_constant_part=preconditioner_constant_part,
                                                            preconditioner_variable_part=preconditioner_variable_part)


    P_k_h = Z_k_h
    P_k_v = Z_k_v
    P_k_U = Z_k_U

    rho_k = torch.sum(R_k_h * Z_k_h) + torch.sum(R_k_v * Z_k_v) + torch.sum(R_k_U * Z_k_U)

    norm_r_0 = torch.sqrt(torch.sum(R_k_h**2) + torch.sum(R_k_v**2) + torch.sum(R_k_U**2))

    for iteration_CG in range(1, int(max_iter_CG) + 1):

        linear_map_k_h, linear_map_k_v, linear_map_k_U = apply_linear_map(P_k_h, P_k_v, P_k_U, Qh, Qv, tau, S, T)

        out = torch.trace(P_k_h.T @ linear_map_k_h) + torch.trace(P_k_v.T @ linear_map_k_v) + torch.trace(P_k_U.T @ linear_map_k_U)
        alpha_k = rho_k / out

        Vh = Vh + (alpha_k * P_k_h)
        Vv = Vv + (alpha_k * P_k_v)
        U = U + (alpha_k * P_k_U)

        R_k1_h = R_k_h - (alpha_k * linear_map_k_h)
        R_k1_v = R_k_v - (alpha_k * linear_map_k_v)
        R_k1_U = R_k_U - (alpha_k * linear_map_k_U)



        Z_k1_h, Z_k1_v, Z_k1_U = apply_preconditioner(R_k1_h, R_k1_v, R_k1_U, tau=tau,
                                                                    preconditioner_name=preconditioner_name,
                                                                    preconditioner_constant_part=preconditioner_constant_part,
                                                                    preconditioner_variable_part=preconditioner_variable_part)

        rho_k1 = torch.sum(R_k1_h * Z_k1_h) + torch.sum(R_k1_v * Z_k1_v) + torch.sum(R_k1_U * Z_k1_U)
        beta = rho_k1 / rho_k

        P_k1_h = Z_k1_h + (beta * P_k_h)
        P_k1_v = Z_k1_v + (beta * P_k_v)
        P_k1_U = Z_k1_U + (beta * P_k_U)

        # updates
        R_k_h = R_k1_h
        R_k_v = R_k1_v
        R_k_U = R_k1_U

        P_k_h = P_k1_h
        P_k_v = P_k1_v
        P_k_U = P_k1_U

        rho_k = rho_k1

        Z_k_h = Z_k1_h
        Z_k_v = Z_k1_v
        Z_k_U = Z_k1_U

        residue_norm = torch.sqrt(torch.sum(R_k_h**2) + torch.sum(R_k_v**2) + torch.sum(R_k_U**2))
        relative_residue_norm = residue_norm / norm_r_0

        if iteration_CG % (max_iter_CG / 10) == 0:
            if verbose:
                print("    CG iteration", iteration_CG, "residue norm =", residue_norm.item(), "relative residue norm =", relative_residue_norm.item())

        if residue_norm < abs_tol or relative_residue_norm < rel_tol:
            if verbose:
                print("    Breaking at CG iteration", iteration_CG, "residue norm =", residue_norm.item(), "relative residue norm =", relative_residue_norm.item())
            break

    return Vh, Vv, U



def apply_linear_map(Vh, Vv, U, Qh, Qv, tau, S, T):
    SU_minus_Vh = 1/tau * (S @ U - Vh)
    UT_minus_Vv = 1/tau * (U @ T - Vv)
    top_part = Qh * Vh - SU_minus_Vh
    middle_part = Qv * Vv - UT_minus_Vv
    bottom_part = S.T @ SU_minus_Vh
    bottom_part = bottom_part + UT_minus_Vv @ T.T
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

def get_preconditioner(preconditioner_name, S, T, Qh, Qv, tau):
    if preconditioner_name == "None":
        return None
    elif preconditioner_name == "block_diag" or preconditioner_name == "block_diag_dct":
        Vh_part = Qh + 1/tau 
        Vv_part = Qv + 1/tau 

        return (Vh_part, Vv_part)


def apply_preconditioner(R_h, R_v, R_U, tau, preconditioner_variable_part,
                                      preconditioner_constant_part, preconditioner_name):
    if preconditioner_name == "None":
        return R_h, R_v, R_U
    elif preconditioner_name == "block_diag":
        Vh_part, Vv_part = preconditioner_variable_part

        PS, PS_transpose, PT, PT_transpose, kron_diag = preconditioner_constant_part

        res_h = R_h / Vh_part
        res_v = R_v / Vv_part

        res_U = solve_sylvester(R_U, PS, PS_transpose, PT, PT_transpose, kron_diag, tau)
        return res_h, res_v, res_U

    elif preconditioner_name == "block_diag_dct":
        Vh_part, Vv_part = preconditioner_variable_part

        kron_diag = preconditioner_constant_part

        res_h = R_h / Vh_part
        res_v = R_v / Vv_part


        RHS = dct.dct_2d(tau*R_U)
        PUP = RHS / kron_diag
        res_U = dct.idct_2d(PUP)
        return res_h, res_v, res_U 


def solve_sylvester(R, PS, PS_transpose, PT, PT_transpose, kron_diag, tau):
    RHS = PS_transpose @ (tau * R) @ PT
    PUP = RHS / kron_diag

    Res = PS @ PUP @ PT_transpose
    return Res



def F_delta(delta, Vh, Vv, U, Wh, Wv, tau, Gh, Gv, S, T, Ch=0, Cv=0):
    term_h = 0.5 * torch.sum( ((Ch * Vh)**2 + delta**2) * (1.0 / Wh) +  Wh)
    term_v = 0.5 * torch.sum( ((Cv * Vv)**2 + delta**2) * (1.0 / Wv) +  Wv)
    term_reg = 0.5 / tau * ( torch.sum( (Vh - S@U + Gh)**2) + torch.sum( (Vv - U@T + Gv)**2))
    return term_h + term_v + term_reg