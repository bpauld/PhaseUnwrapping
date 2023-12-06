import torch
import numpy as np

def wrap_matrix(X):
    Z = X - 2 * np.pi * np.floor((X + np.pi) / (2 * np.pi))
    return Z

def build_S(N):
    S = np.zeros((N - 1, N))
    for i in range(N - 1):
        S[i, i + 1] = 1
        S[i, i] = -1
    return S

def build_S_transpose(N):
    S = torch.zeros(N - 1, N)
    for i in range(N - 1):
        S[i, i + 1] = 1
        S[i, i] = -1
    return S.T

def build_T(M):
    T = np.zeros((M, M - 1))
    for i in range(M - 1):
        T[i + 1, i] = 1
        T[i, i] = -1
    return T

def build_T_transpose(M):
    T = torch.zeros(M, M - 1)
    for i in range(M - 1):
        T[i + 1, i] = 1
        T[i, i] = -1
    return T.T

def apply_S(U):
    return U[1:, :] - U[:-1, :]

def apply_T(U):
    return U[:, 1:] - U[:, :-1]

def apply_T_transpose(UT):
    N, M1 = UT.size()
    M = M1 + 1
    res = torch.zeros(N, M)
    res[:, 0] = -UT[:, 0]
    res[:, 1:M-1] = -UT[:, 1:M-1] + UT[:, 0:M-2]
    res[:, -1] = UT[:, -1]
    return res

def apply_S_transpose(SU):
    N1, M = SU.size()
    N = N1 + 1
    res = torch.zeros(N, M)
    res[0, :] = -SU[0, :]
    res[1:N-1, :] = -SU[1:N-1, :] + SU[0:N-2, :]
    res[-1, :] = SU[-1, :]
    return res
