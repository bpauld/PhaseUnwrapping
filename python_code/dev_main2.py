import torch
from image_manipulation import *
from utils import get_diagonal_kron_identity
import torch_dct as dct


DEFAULT_TYPE = torch.float32

def get_preconditioner2(preconditioner_name, S, T, Wh, Wv):

    return None
    #if preconditioner_name == "None":
    #    return None
    #elif preconditioner_name == "block_diag" or preconditioner_name == "block_diag_dct":
    #    Vh_part = Wh + 1/tau 
    #    Vv_part = Wv + 1/tau 

     #   return (Vh_part, Vv_part)


def apply_preconditioner2(R, preconditioner_variable_part,
                                      preconditioner_constant_part, preconditioner_name):
    if preconditioner_name == "None":
        return R
    elif preconditioner_name == "block_diag":
        #Vh_part, Vv_part = preconditioner_variable_part

        PS, PS_transpose, PT, PT_transpose, kron_diag = preconditioner_constant_part

        res = solve_sylvester2(R, PS, PS_transpose, PT, PT_transpose, kron_diag)
        return res

    elif preconditioner_name == "block_diag_dct":
        kron_diag = preconditioner_constant_part

        RHS = dct.dct_2d(R)
        PUP = RHS / kron_diag
        res = dct.idct_2d(PUP)
        return res

@torch.jit.script
def solve_sylvester2(R, PS, PS_transpose, PT, PT_transpose, kron_diag):
    RHS = PS_transpose @ (R) @ PT
    PUP = RHS / kron_diag

    Res = PS @ PUP @ PT_transpose
    return Res

def IRLS2(X, Ch=None, Cv=None, epsilon=1, max_iter=1e4,
                      max_iter_CGM_start=5, max_iter_CGM=1000, phi=0.1, alpha=0.5,
                      approx_minimization_method="CG", preconditioner_name="None",
                      abs_tol_CGM=1e-10, rel_tol_CGM=0, rel_improvement_tol=1e-3,
                      increase_CGM_iteration_factor=1.7, verbose=True):
    N, M = X.shape

    S = build_S(N)
    T = build_T(M)
    Gh = wrap_matrix(apply_S(X))
    Gv = wrap_matrix(apply_T(X))

    Gh = torch.tensor(Gh, dtype=DEFAULT_TYPE)
    Gv = torch.tensor(Gv, dtype=DEFAULT_TYPE)
    S = torch.tensor(S, dtype=DEFAULT_TYPE)
    T = torch.tensor(T, dtype=DEFAULT_TYPE)

    if Ch is None:
        Ch = torch.ones((N-1, M), dtype=DEFAULT_TYPE)
        Cv = torch.ones((N, M-1), dtype=DEFAULT_TYPE)
    else:
        Ch = torch.tensor(Ch, dtype=DEFAULT_TYPE)
        Cv = torch.tensor(Cv, dtype=DEFAULT_TYPE)

    Wh = torch.ones((N-1, M), dtype=DEFAULT_TYPE)
    Wv = torch.ones((N, M-1), dtype=DEFAULT_TYPE)
    U = torch.zeros((N, M), dtype=DEFAULT_TYPE)

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

    preconditioner_constant_part = None
    DS = None
    PS = None
    DT = None
    PT = None
    kron_diag = None

    if preconditioner_name == "block_diag":
        DS, PS = torch.linalg.eigh((S.T @ S).to_dense(), UPLO="L")
        DS = torch.where(DS > 1e-4, DS, 1e-4)

        print(DS)

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
        DS = DS.cuda()
        DT = DT.cuda()
        PS = PS.cuda()  # Assuming you're using CUDA
        PT = PT.cuda()  # Assuming you're using CUDA
        PS_transpose = PS.T.cuda()  # Assuming you're using CUDA
        PT_transpose = PT.T.cuda()  # Assuming you're using CUDA
        kron_diag = kron_diag.cuda()  # Assuming you're using CUDA
        preconditioner_constant_part = (PS, PS_transpose, PT, PT_transpose, kron_diag)

    if preconditioner_name == "block_diag_dct":
        DS = torch.diag(dct.dct_2d((S.T @ S).to_dense()))
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
        kron_diag = kron_diag.cuda()  # Assuming you're using CUDA
        preconditioner_constant_part = kron_diag


    epsilon = torch.tensor(epsilon, dtype=DEFAULT_TYPE).cuda()  # Assuming you're using CUDA


    Wh = 1.0 / torch.sqrt(Ch**2 * (S @ U - Gh)**2 + epsilon**2)
    Wv = 1.0 / torch.sqrt(Cv**2 * (U @ T - Gv)**2 + epsilon**2)

    J_eps_prev = J_epsilon2(epsilon=epsilon, U=U, Wh=Wh, Wv=Wv, Gh=Gh, Gv=Gv, S=S, T=T, Ch=Ch, Cv=Cv)
    if verbose:
        print("J_epsilon at start =", J_eps_prev)
    
    just_increased_max_iter_CGM = False

    next_max_iter_CGM = max_iter_CGM_start
    stop_at_next = False
    stepsize = epsilon

    for iteration in range(int(max_iter)):
        if verbose:
            print("########################### Iteration", iteration, ", epsilon =", epsilon, "#####################################")
            print(torch.sum(U).item())

        U_prev = U

        if verbose:
            print("Maximum number of CGM iterations = ", next_max_iter_CGM)
        U = CG_2(Ch**2 * Wh, Cv**2 * Wv, S, T,
                                        Gh, Gv, U,
                                        max_iter_CGM=next_max_iter_CGM,
                                        preconditioner_name=preconditioner_name,
                                        preconditioner_constant_part=preconditioner_constant_part,
                                        abs_tol=abs_tol_CGM, rel_tol=rel_tol_CGM,
                                        verbose=verbose)

        if verbose:
            print("Sum(U) = ", torch.sum(U).item())
        U = U - torch.mean(U)
        J_eps_new = J_epsilon2(epsilon=epsilon, U=U, Wh=Wh, Wv=Wv, Gh=Gh, Gv=Gv, S=S, T=T, Ch=Ch, Cv=Cv)
        relative_improvement = (J_eps_prev - J_eps_new) / J_eps_prev
        if verbose:
            print("J_epsilon after updating Vh, Vv, U =", J_eps_new.item(), ", relative improvement =", relative_improvement.item())

        J_eps_prev = J_eps_new

        # Update weights
        Wh = 1.0 / torch.sqrt(Ch**2 * (S @ U - Gh)**2 + epsilon**2)
        Wv = 1.0 / torch.sqrt(Cv**2 * (U @ T - Gv)**2 + epsilon**2)

        J_eps_new = J_epsilon2(epsilon=epsilon, U=U, Wh=Wh, Wv=Wv, Gh=Gh, Gv=Gv, S=S, T=T, Ch=Ch, Cv=Cv)
        relative_improvement = (J_eps_prev - J_eps_new) / J_eps_prev
        if verbose:
            print("J_epsilon after updating weights =", J_eps_new.item(), ", relative_improvement =", relative_improvement.item())

        J_eps_prev = J_eps_new

        if stop_at_next:
            break
        if relative_improvement < rel_improvement_tol:
            if just_increased_max_iter_CGM:
                break
            else:
                next_max_iter_CGM = min(int(np.round(next_max_iter_CGM * increase_CGM_iteration_factor)), max_iter_CGM)
                just_increased_max_iter_CGM = True
        else:
            just_increased_max_iter_CGM = False

    return U.cpu().numpy()


def CG_2(Wh, Wv, S, T, Gh, Gv, U_start,
                        max_iter_CGM, preconditioner_name, preconditioner_constant_part,
                        abs_tol=1e-10, rel_tol=0, verbose=True):

    N = S.shape[0]
    M = T.shape[1]

    U = U_start

    preconditioner_variable_part = get_preconditioner2(preconditioner_name=preconditioner_name,
                                                                    S=S, T=T, Wh=Wh, Wv=Wv)


    B = S.T @ ( Wh * Gh) + (Wv * Gv) @ T.T

    linear_map_0 = apply_linear_map2(U_start, Wh, Wv, S, T)
    R_k = B - linear_map_0
    #print(torch.sum(torch.abs(B_U)), torch.sum(torch.abs(linear_map_0_U)))
    #print("Sum (abs(R_k_U)) = ", torch.sum(torch.abs(R_k_U)))

    Z_k = apply_preconditioner2(R_k,
                                  preconditioner_name=preconditioner_name,
                                  preconditioner_constant_part=preconditioner_constant_part,
                                  preconditioner_variable_part=preconditioner_variable_part)


    P_k = Z_k

    rho_k = torch.sum(R_k * Z_k)

    norm_r_0 = torch.sqrt(torch.sum(R_k**2))

    for iteration_CGM in range(1, int(max_iter_CGM) + 1):

        linear_map_k= apply_linear_map2(P_k, Wh, Wv, S, T)

        out = torch.trace(P_k.T @ linear_map_k)
        alpha_k = rho_k / out

        U = U + (alpha_k * P_k)

        R_k1 = R_k - (alpha_k * linear_map_k)

        Z_k1 = apply_preconditioner2(R_k1,
        preconditioner_name=preconditioner_name,
        preconditioner_constant_part=preconditioner_constant_part,
        preconditioner_variable_part=preconditioner_variable_part)

        rho_k1 = torch.sum(R_k1 * Z_k1) 
        beta = rho_k1 / rho_k

        P_k1 = Z_k1 + (beta * P_k)

        # updates
        R_k = R_k1
        P_k = P_k1
        rho_k = rho_k1
        Z_k = Z_k1

        residue_norm = torch.sqrt(torch.sum(R_k**2))
        relative_residue_norm = residue_norm / norm_r_0

        if iteration_CGM % (max_iter_CGM / 10) == 0:
            if verbose:
                print("    CGM iteration", iteration_CGM, "residue norm =", residue_norm.item(), "relative residue norm =", relative_residue_norm.item())

        if residue_norm < abs_tol or relative_residue_norm < rel_tol:
            if verbose:
                print("    Breaking at CGM iteration", iteration_CGM, "residue norm =", residue_norm.item(), "relative residue norm =", relative_residue_norm.item())
            break

    return U



def apply_linear_map2(U, Wh, Wv, S, T):

    return S.T @ ( Wh * (S @ U)) + (Wv * (U @ T)) @ T.T 


def J_epsilon2(epsilon, U, Wh, Wv, Gh, Gv, S, T, Ch=0, Cv=0):

    term_h = 0.5 * torch.sum( ((Ch * (S@U - Gh))**2 + epsilon**2) * Wh + (1.0 / Wh))
    term_v = 0.5 * torch.sum( ((Cv * (U@T - Gv))**2 + epsilon**2) * Wv + (1.0 / Wv))
    #term_psi = 0.5 * sum( (gamma.^2 .* Vpsi.^2 .+ epsilon^2) .* Wpsi .+ (1.0 ./ Wpsi))

    return term_h + term_v 