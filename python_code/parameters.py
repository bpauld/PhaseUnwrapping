from dataclasses import dataclass

@dataclass
class ModelParameters():
    tau: float = 1e-2                              # scalar associated to quadratic penalty term
    delta: float = 1e-6                            # scalar to ensure no divisions by 0 happen.

@dataclass 
class IrlsParameters():
    max_iter: int = 200                             # maximum number of IRLS iterations

    max_iter_CG_strategy: str = "heuristics"        # strategy for updating to maximum number of CG iterations.
    max_iter_CG: int = 1000                         # maximum number of CG iterations.
    max_iter_CG_start: int = 5                      # maximum number of CG iterations in the first iteration of IRLS. This is used only when max_iter_CG_strategy is set to "heuristics"

    rel_improvement_tol: float = 1e-3               # tolerance for relative improvement when updating the maximum number of CG iterations in the heuristics strategy
    increase_CG_max_iteration_factor: float = 1.7   # factor by which the maximum number of CG iterations is multiplied when in the heuristics strategy

    preconditioner: str = "block_diag"              # name of preconditioner. Should be either "None" or "block_diag"

    abs_tol_CG: float = 1e-5                        # absolute tolerence of CG, i.e. CG stops when norm of residue is smaller than this value.
    rel_tol_CG: float = 0.0                         # relative tolerance of CG, i.e. CG stops when relative norm of residue is smaller than this value.