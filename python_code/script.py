import numpy as np
from numpy.linalg import norm
import time
from pathlib import Path
#from dev7 import *
from utils import *
from unwrap import *
from parameters import ModelParameters, IrlsParameters
#from dev_main2 import *



def run_exp2():
    version = "v2"
    size_image = str(2048)
    noise_level = "real_goldstein"

    location = "arz_lebanon"

    path_data = f"/scratch/bpauldub/data/topo_phase_dataset_{version}/data_{size_image}/{location}/npy_files/"

    # specify where you want to save results
    path_results = f"/scratch/bpauldub/results/results_irls_unwrap_{version}/data_{size_image}/{location}/{noise_level}/"

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
        os.mkdir(path_results)
    np.save(os.path.join(path_results, "U.npy"), U)
    np.save(os.path.join(path_results, "duration.npy"), duration)






def run_exp():
    locations = ["arz_lebanon", "etna", "elcapitan",  
                 "kilimanjaro", "mount_sinai", "korab_northmacedonia", "nevada_usa",  "zeil_australia", "wulonggou_china",
                 "warjan_afghanistan"]

    version = "v2"
    size_image = "v3"
    size_image = str(2048)

    noise_level = "real_goldstein"

    preconditioner_name = "block_diag"
    approx_minimization_method = "CG"
    max_iter = 200
    max_iter_CG = 1000
    max_iter_CG_start = 5
    rel_improvement_tol = 1e-3
    increase_CG_iteration_factor = 1.7
    verbose = True
    save_results = True

    tau = 1e-2
    delta=1e-6

    mode = "tau_mode"

    rel_tol_CG = 0
    abs_tol_CG = 1e-5

    for location in locations:
        print(location)
        my_func7(size_image=size_image, 
                 approx_minimization_method=approx_minimization_method,
                 max_iter_CG=max_iter_CG, 
                 max_iter_CG_start=max_iter_CG_start,
                 rel_improvement_tol=rel_improvement_tol,
                 increase_CG_iteration_factor=increase_CG_iteration_factor,
                 delta=delta, 
                 tau=tau, preconditioner_name=preconditioner_name, 
                 rel_tol_CG=rel_tol_CG,
                 abs_tol_CG=abs_tol_CG,
                 noise_level=noise_level, version=version,
                 save_results=save_results,
                 max_iter=max_iter, location=location,
                 mode=mode,
                 verbose=verbose)

def my_func7(size_image, 
             approx_minimization_method="CG",
             max_iter_CG=1,
             max_iter_CG_start=5,
             rel_improvement_tol=1e-3,
             increase_CG_iteration_factor=1.7,
             tau=1, delta=1e-6, 
             max_iter=10, preconditioner_name="None", location="etna",
             save_results=True, noise_level="noiseless", version="v1",
             mode="tau_mode",
             abs_tol_CG=1e-3, rel_tol_CG=1e-3,  verbose=True):

    path_data = f"/scratch/bpauldub/data/topo_phase_dataset_{version}/data_{size_image}/"

    path_data2 = os.path.join(path_data, location, "npy_files")


    amp1 = None
    amp2 = None
    corrfile = None
    if os.path.isfile(os.path.join(path_data2, "amp1.npy")) and os.path.isfile(os.path.join(path_data2, "amp2.npy")):
        amp1 = np.load(os.path.join(path_data2, "amp1.npy")).astype(np.float32)
        amp2 = np.load(os.path.join(path_data2, "amp2.npy")).astype(np.float32) 
        print("Found amplitude files")
    else:
        print("Did not find amplitude files")
    if os.path.isfile(os.path.join(path_data2, "coherence.npy")):
        corrfile = np.load(os.path.join(path_data2, "coherence.npy")).astype(np.float32)
        print("Found coherence file")
    else:
        print("Did not find coherence file")

    path_weights = f"/scratch/bpauldub/results/results_snaphu_unwrap_{version}/data_{size_image}/{location}/{noise_level}/mcfL1"
    #path_save = f"/scratch/bpauldub/results/results_irls_unwrap_{version}/data_{size_image}/{location}/{noise_level}/"

    X, X_real, X_unw, X_nunw, X_sigma_noise = load_data(location, path=path_data, noise_level=noise_level)

    N, M = X.shape

    tmpdir = "/home/bpauldub/tmp_snaphu"
    Ch_snaphu, Cv_snaphu = get_weights_from_snaphu(X.astype(np.float32), tmpdir, amp1, amp2, corrfile)




    Ch = np.ones((N-1, M))
    Cv = np.ones((N, M-1))

    print(X.shape)

    if Path(f"{path_weights}/c1.npy").is_file():
        Ch, Cv = load_weights(path_weights)
        print("Found weight files.")
    else:
        print("No weight file found. Running with uniform weights.")

    print(np.linalg.norm(Ch - Ch_snaphu), np.linalg.norm(Cv - Cv_snaphu))

    print(np.max([np.max(Ch), np.max(Cv)]))

    Cv[0, 0] = np.median(Cv)
    Cv[0, -1] = np.median(Cv)
    Cv[-1, 0] = np.median(Cv)
    Cv[-1, -1] = np.median(Cv)

    Ch = Ch.astype(float)
    Cv = Cv.astype(float)

    max_value = np.max([np.max(Ch), np.max(Cv)])
    print(max_value)


    Ch /= float(max_value)
    Cv /= float(max_value)
  
    duration = 0

    if mode=="tau_mode":
        start_time = time.time()
        U, Vh, Vv = IRLS(X, tau=tau,
                                            Ch=Ch, Cv=Cv,
                                            delta=delta, max_iter=max_iter, 
                                            approx_minimization_method=approx_minimization_method,
                                            max_iter_CG=max_iter_CG,
                                            preconditioner_name=preconditioner_name,
                                            abs_tol_CG=abs_tol_CG, rel_tol_CG=rel_tol_CG,
                                            max_iter_CG_start=max_iter_CG_start,
                                            rel_improvement_tol=rel_improvement_tol,
                                            increase_CG_iteration_factor=increase_CG_iteration_factor, verbose=verbose)
        duration = time.time() - start_time
        print(duration)
        Gh = wrap_matrix(apply_S(X)); Gv = wrap_matrix(apply_T(X))
        U_reconstructed_with_gradients = reconstruct_image_from_gradients(Vh + Gh, Vv + Gv)
    else:
        start_time = time.time()
        U = IRLS2(X,
                                            Ch=Ch, Cv=Cv,
                                            delta_start=delta, max_iter=max_iter, 
                                            approx_minimization_method=approx_minimization_method,
                                            max_iter_CG=max_iter_CG,
                                            preconditioner_name=preconditioner_name,
                                            abs_tol_CG=abs_tol_CG, rel_tol_CG=rel_tol_CG,
                                            max_iter_CG_start=max_iter_CG_start,
                                            rel_improvement_tol=rel_improvement_tol,
                                            increase_CG_iteration_factor=increase_CG_iteration_factor, verbose=verbose)
        duration = time.time() - start_time
        U_reconstructed_with_gradients = U
        print(duration)

    
    error = compute_error(U, X_unw)
    error_noisy = compute_error(U, X_nunw)

    
    norm_error_with_reconstructed = norm(compute_error(U, U_reconstructed_with_gradients))
    print(f"Error between output and output reconstructed with gradients = {norm_error_with_reconstructed}")

    if save_results and (N > 512):
        save_results_npy(U, U_reconstructed_with_gradients, X_unw, X_nunw, X, error, error_noisy, X_real, X_sigma_noise, duration=duration, path=path_save)


if __name__ == "__main__":
    run_exp2()