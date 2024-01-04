import numpy as np
import time
from pathlib import Path
from python_code.utils import *
from python_code.unwrap import unwrap
from python_code.parameters import ModelParameters, IrlsParameters


def run_exp():
    location = "arz_lebanon"
    print(location)
    
    #specify path to data here
    path_data = os.path.join(os.getcwd(), "data", location, "npy_files")

    # Define model parameters 
    tau = 1e-2
    delta = 1e-6
    model_params = ModelParameters()
    model_params.tau = tau
    model_params.delta = delta

    # run on gpu
    run_on_gpu = False

    # Define weighting strategy.
    # Here you can define or load your own weights Ch and Cv.
    Cv = None
    Ch = None
    # If the above weights are not set to None,
    # setting weighting_strategy to "snaphu_weights" will compute statistical-based weights using SNAPHU,
    # otherwise setting weighting_strategy to None or to "uniform" will set all weights equal to 1
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



    # First load simulated image
    path_X = os.path.join(path_data,  "simu_unwrapped.npy")
    X = wrap_matrix(np.load(path_X))

    # For simulated image, we do not use amplitude and correlation files
    amp1 = None
    amp2 = None
    corrfile = None    

    start_time = time.time()
    U, Vv, Vh = unwrap(X,
    model_params=model_params,
    irls_params=irls_params,
    amp1=amp1, amp2=amp2, corrfile=corrfile,
    weighting_strategy=weighting_strategy,
    Cv=Cv, Ch=Ch, snaphu_config_file=snaphu_config_file, snaphu_bin=snaphu_bin,
    run_on_gpu=run_on_gpu,
    verbose=True)

    duration = time.time() - start_time
    print("Total duration (s) = ", duration)    
    
    # save results
    path_results = os.path.join(os.getcwd(), "results", location, "noiseless")
    if not os.path.isdir(path_results):
        os.makedirs(path_results)
    np.save(os.path.join(path_results, "U.npy"), U)
    np.save(os.path.join(path_results, "duration.npy"), duration)


    # Now load real image
    path_X = os.path.join(path_data,  "real_goldstein.npy")
    X = np.load(path_X)
    
    # For real images, we use amplitude and correlation files
    path_amp1 = os.path.join(path_data,  "amp1.npy")
    path_amp2 = os.path.join(path_data, "amp2.npy")
    path_corrfile = os.path.join(path_data, "coherence.npy")
    amp1 = np.load(path_amp1)
    amp2 = np.load(path_amp2)
    corrfile = np.load(path_corrfile)

    start_time = time.time()
    U, Vv, Vh = unwrap(X,
    model_params=model_params,
    irls_params=irls_params,
    amp1=amp1, amp2=amp2, corrfile=corrfile,
    weighting_strategy=weighting_strategy,
    Cv=Cv, Ch=Ch, snaphu_config_file=snaphu_config_file, snaphu_bin=snaphu_bin,
    run_on_gpu=run_on_gpu,
    verbose=True)

    duration = time.time() - start_time
    print("Total duration (s) = ", duration)    
    
    # save results
    path_results = os.path.join(os.getcwd(), "results", location, "real_goldstein")
    if not os.path.isdir(path_results):
        os.makedirs(path_results)
    np.save(os.path.join(path_results, "U.npy"), U)
    np.save(os.path.join(path_results, "duration.npy"), duration)



if __name__ == "__main__":
    run_exp()
