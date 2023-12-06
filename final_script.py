import numpy as np
import time
from pathlib import Path
from python_code.utils import *
from python_code.unwrap import unwrap
from python_code.parameters import ModelParameters, IrlsParameters


def run_exp():
    version = "v2"
    size_image = str(2048)
    noise_level = "real_goldstein"

    locations = ["arz_lebanon", "etna", "elcapitan",  
                 "kilimanjaro", "mount_sinai", "korab_northmacedonia", "nevada_usa",  "zeil_australia", "wulonggou_china",
                 "warjan_afghanistan"]


    #specify path to data here
    #path_data = f"/scratch/bpauldub/data/topo_phase_dataset_{version}/data_{size_image}/{location}/npy_files/"
    path_data = os.path.join(os.getcwd(), "data", "arz_lebanon", "npy_files")

    # specify where you want to save results
    #path_results = f"/scratch/bpauldub/results/results_irls_unwrap_{version}/data_{size_image}/{location}/{noise_level}/"
    path_results = os.path.join(os.getcwd(), "results", "arz_lebanon")

    # load your data here
    path_X = os.path.join(path_data,  f"{noise_level}.npy")
    path_amp1 = os.path.join(path_data, "amp1.npy")
    path_amp2 = os.path.join(path_data, "amp2.npy")
    path_corr = os.path.join(path_data, "coherence.npy")
    X = np.load(path_X)
    if os.path.isfile(path_amp1) and os.path.isfile(path_amp2):
        amp1 = np.load(path_amp1)
        amp2 = np.load(path_amp2)
    else:
        amp1 = None
        amp2 = None
    if os.path.isfile(path_corr):
        corrfile = np.load(path_corr)
    else:
        corrfile = None    


    # Define model parameters 
    tau = 1e-2
    delta = 1e-6
    model_params = ModelParameters()
    model_params.tau = tau
    model_params.delta = delta

    # Define weighting strategy.
    # Here you can define or load your own weights Ch and Cv.
    Ch = None
    Cv = None
    # If the above weights are not set to None,
    # setting weighting_strategy to "snaphu_weights" will compute statistical-based weights using SNAPHU,
    # otherwise setting weighting_strategy to None will set all weights equal to 1
    weighting_strategy = "snaphu_weights"

    # Must specify a path to SNAPHU if weighting_strategy is "snaphu_weights"
    snaphu_bin = os.path.join(os.getcwd(), "snaphu-v2.0.6/bin/snaphu")
    # Can specify a SNAPHU configuration file to compute weights.
    # It must be in the format specified in the SNAPHU documentation (see "https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu.conf.full")
    # If not specified, default SNAPHU parameters are used.
    snaphu_config_file = None

    # Define algorithm parameters
    preconditioner_name = "block_diag"
    max_iter_CG_strategy = "heuristics"
    max_iter = 200
    max_iter_CG = 1000
    max_iter_CG_start = 5
    rel_improvement_tol = 1e-3
    increase_CG_iteration_factor = 1.7
    rel_tol_CG = 0
    abs_tol_CG = 1e-5
    irls_params = IrlsParameters()
    irls_params.max_iter = max_iter
    irls_params.max_iter_CG = max_iter_CG
    irls_params.max_iter_CG_strategy = max_iter_CG_strategy
    irls_params.max_iter_CG_start = max_iter_CG_start
    irls_params.preconditioner = preconditioner_name
    irls_params.rel_improvement_tol = rel_improvement_tol
    irls_params.increase_CG_max_iteration_factor = increase_CG_iteration_factor
    irls_params.rel_tol_CG = rel_tol_CG
    irls_params.abs_tol_CG = abs_tol_CG


    start_time = time.time()
    U, Vh, Vv = unwrap(X,
    model_params=model_params,
    irls_params=irls_params,
    amp1=amp1, amp2=amp2, corrfile=corrfile,
    weighting_strategy=weighting_strategy,
    Ch=Ch, Cv=Cv, snaphu_config_file=snaphu_config_file, snaphu_bin=snaphu_bin,
    verbose=True)

    duration = time.time() - start_time
    print("Total duration (s) = ", duration)


    # save results
    if not os.path.isdir(path_results):
        os.makedirs(path_results)
    np.save(os.path.join(path_results, "U.npy"), U)
    np.save(os.path.join(path_results, "duration.npy"), duration)



if __name__ == "__main__":
    run_exp()