import torch
from python_code.image_manipulation import *
import torch_dct as dct
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
    verbose=True):
    """Unwraps an image X.

    Arguments:
        X: ndarray
            array of shape (N, M) containing image to be unwrapped
        model_params: ModelParameters, optional
            parameters of the model (default is ModelParameters())
        irls_params: IrlsParams, optional
            parameters for running the IRLS algorithm (default is IrlsParameters())
        amp1: ndarray, optional
            amplitude of the first image in the interferogram. Is used to compute weights if available (default is None)
        amp2: ndarray, optional
            amplitude of the second image in the interferogram. Is used to compute weights if available (default is None)
        corrfile: ndarray, optional
            coherence map of the interferogram. Is used to compute weights if available (default is None)
        Ch: ndarray, optional.
            user-supplied weights of shape (N-1, M) (default is None).
        Cv: ndarray, optional.
            user-supplied weights of shape (N, M-1) (default is None).
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
        U: ndarray
            unwrapped image ndarray of shape (N, M)
    """

    N, M = X.shape

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
            raise RuntimeError("Cannot find snaphu bin at specified location {}".format(snaphu_bin))
        else:
            print("Computing statistical-based weights using SNAPHU")
            tmpdir = os.path.join(os.getcwd(), "snaphu_weights_tmp_dir") #will be removed anyway
            Ch, Cv = get_weights_from_snaphu(X, tmpdir, amp1=amp1, amp2=amp2, corrfile=corrfile, snaphu_bin=snaphu_bin, snaphu_config_file=snaphu_config_file)
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


    U, Vh, Vv = IRLS(X, Ch, Cv, model_params=model_params, irls_params=irls_params, verbose=verbose)

    return U, Vh, Vv



def IRLS(X, Ch, Cv, model_params: ModelParameters=ModelParameters(),
        irls_params: IrlsParameters=IrlsParameters(), verbose=True):
    
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
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found. Running on CPU. This will significantly impact performance")

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

    if torch.cuda.is_available():
        S = S.cuda() 
        T = T.cuda() 
        U = U.cuda()  
        Gh = Gh.cuda() 
        Gv = Gv.cuda() 
        Wh = Wh.cuda()  
        Wv = Wv.cuda() 
        Ch = Ch.cuda()
        Cv = Cv.cuda()


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
            #DT = torch.where(TT_eigenvalues > 1e-6, TT_eigenvalues, 1e-6)

        if verbose:
            print("Done computing decomposition of S'S and of TT'...")

        kron_diag = get_diagonal_kron_identity(DS.cpu().numpy(), "left", M) + get_diagonal_kron_identity(DT.cpu().numpy(), "right", N)
        #kron_diag = torch.kron(torch.eye(M), torch.diag(DS)) + torch.kron(torch.diag(DT), torch.eye(N))

        kron_diag = kron_diag.reshape((N, M), order='F')
        kron_diag = torch.tensor(kron_diag, dtype=DEFAULT_TYPE)

        PS_transpose = PS.T
        PT_transpose = PT.T

        if torch.cuda.is_available():
            DS = DS.cuda()
            DT = DT.cuda()
            PS = PS.cuda() 
            PT = PT.cuda() 
            PS_transpose = PS_transpose.cuda() 
            PT_transpose = PT_transpose.cuda()  
            kron_diag = kron_diag.cuda()  
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
        if torch.cuda.is_available():
            kron_diag = kron_diag.cuda()
        preconditioner_constant_part = kron_diag




    tau = torch.tensor(tau, dtype=DEFAULT_TYPE)
    delta = torch.tensor(delta, dtype=DEFAULT_TYPE)
    if torch.cuda.is_available():
        tau = tau.cuda()
        delta = delta.cuda()

    Wh = 1.0 / torch.sqrt(Ch**2 * Vh**2 + delta**2)
    Wv = 1.0 / torch.sqrt(Cv**2 * Vv**2 + delta**2)

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
            print("########################### Iteration", iteration, ", delta =", delta, "#####################################")
            print(torch.sum(U).item())

        # Update Vh, Vv, Vpsi with CG
        Vh_prev = Vh
        Vv_prev = Vv
        U_prev = U

        if verbose:
            print("Maximum number of CG iterations = ", next_max_iter_CG)
        Vh, Vv, U = CG(Ch**2 * Wh, Cv**2 * Wv, S, T,
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
        Wh = 1.0 / torch.sqrt(Ch**2 * Vh**2 + delta**2)
        Wv = 1.0 / torch.sqrt(Cv**2 * Vv**2 + delta**2)

        F_delta_new = F_delta(delta=delta, Vh=Vh, Vv=Vv, U=U, Wh=Wh, Wv=Wv, tau=tau, Gh=Gh, Gv=Gv, S=S, T=T, Ch=Ch, Cv=Cv)
        relative_improvement = (F_delta_prev - F_delta_new) / F_delta_prev
        if verbose:
            print("F_delta after updating Wh, Wv =", F_delta_new.item(), ", relative_improvement =", relative_improvement.item())

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


def CG(Wh, Wv, S, T, tau, Gh, Gv, Vh_start, Vv_start, U_start,
                        max_iter_CG, preconditioner_name, preconditioner_constant_part,
                        abs_tol=1e-10, rel_tol=0, verbose=True):

    N = S.shape[0]
    M = T.shape[1]

    Vh = Vh_start
    Vv = Vv_start
    U = U_start

    preconditioner_variable_part = get_preconditioner(preconditioner_name=preconditioner_name,
                                                                    S=S, T=T, Wh=Wh, Wv=Wv, tau=tau)

    B_h = -1 / tau * Gh
    B_v = -1 / tau * Gv
    B_U = 1 / tau * (S.T @ Gh + Gv @ T.T)

    linear_map_0_h, linear_map_0_v, linear_map_0_U = apply_linear_map(Vh_start, Vv_start, U_start, Wh, Wv, tau, S, T)
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

        linear_map_k_h, linear_map_k_v, linear_map_k_U = apply_linear_map(P_k_h, P_k_v, P_k_U, Wh, Wv, tau, S, T)

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



def apply_linear_map(Vh, Vv, U, Wh, Wv, tau, S, T):
    SU_minus_Vh = 1/tau * (S @ U - Vh)
    UT_minus_Vv = 1/tau * (U @ T - Vv)
    top_part = Wh * Vh - SU_minus_Vh
    middle_part = Wv * Vv - UT_minus_Vv
    bottom_part = S.T @ SU_minus_Vh
    bottom_part = bottom_part + UT_minus_Vv @ T.T
    return top_part, middle_part, bottom_part  




def get_diagonal_kron_identity(K, side, identity_size):
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

def get_preconditioner(preconditioner_name, S, T, Wh, Wv, tau):

    if preconditioner_name == "None":
        return None
    elif preconditioner_name == "block_diag" or preconditioner_name == "block_diag_dct":
        Vh_part = Wh + 1/tau 
        Vv_part = Wv + 1/tau 

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
    term_h = 0.5 * torch.sum( ((Ch * Vh)**2 + delta**2) * Wh + (1.0 / Wh))
    term_v = 0.5 * torch.sum( ((Cv * Vv)**2 + delta**2) * Wv + (1.0 / Wv))
    term_reg = 0.5 / tau * ( torch.sum( (Vh - S@U + Gh)**2) + torch.sum( (Vv - U@T + Gv)**2))
    return term_h + term_v + term_reg