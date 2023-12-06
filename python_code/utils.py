import numpy as np
import os
from python_code.image_manipulation import *
from typing import Any
import subprocess
import shutil



def get_weights_from_snaphu(input, tmpdir, amp1=None, amp2=None, corrfile=None,
snaphu_bin: str = "/home/bpauldub/PhaseUnwrap/snaphu-v2.0.6/bin/snaphu", 
snaphu_config_file=None):

    tmpdir = os.path.abspath(tmpdir)
    os.makedirs(tmpdir, exist_ok=True)

    assert len(input.shape) == 2, input.shape
    assert input.dtype == np.float32, input.dtype
    h, w = input.shape

    logfile = os.path.join(tmpdir, "log.txt")
    inpath = os.path.join(tmpdir, "input")
    outpath = os.path.join(tmpdir, "output")
    stdout = os.path.join(tmpdir, "stdout.txt")
    costout = os.path.join(tmpdir, "cost_out")
    mstcostsout = os.path.join(tmpdir, "snaphu.mstcosts")

    if amp1 is not None and amp2 is not None:
        amp1path = os.path.join(tmpdir, "amp1")
        amp2path = os.path.join(tmpdir, "amp2")
        amp1.tofile(amp1path)
        amp2.tofile(amp2path)
    else:
        print("Missing amplitude files. Computing SNAPHU weights without amplitude.")
    if corrfile is not None:
        corrpath = os.path.join(tmpdir, "corrfile")
        corrfile.tofile(corrpath)
    else:
        print("Missing correlation file. Computing SNAPHU weights without correlation.")

    params = {
        "INFILEFORMAT": "FLOAT_DATA",
        "OUTFILE": outpath,
        "OUTFILEFORMAT": "FLOAT_DATA",
        "CORRFILEFORMAT": "FLOAT_DATA",
        "UNWRAPPEDINFILEFORMAT": "FLOAT_DATA"
    }

    cliparams = " ".join(f'-C "{k} {v}"' for k, v in params.items())

    cmd = f"{snaphu_bin}"

    if snaphu_config_file is not None:
        cmd += f" -f {snaphu_config_file}"

    cmd += f" {cliparams}"

    input.tofile(inpath)

    cmd += f" -W"

    cmd += f" --dumpall -l {logfile}"
    
    if amp1 is not None and amp2 is not None:
        cmd += f" --aa {amp1path} {amp2path}"
    if corrfile is not None:
        cmd += f" -c {corrpath}"    

    cmd += f" {inpath} {w} 2>&1 | tee {stdout}"
    #print(cmd)
    assert not subprocess.call(cmd, shell=True, cwd=tmpdir)
    c1 = 0
    c2 = 0
    if os.path.isfile(mstcostsout):
        cost_out = np.fromfile(mstcostsout, dtype=np.int16)
        c1, c2 = reshape_weights(cost_out, h, w)
    # remove temp dir
    shutil.rmtree(tmpdir)
    return c1, c2


def reshape_weights(costs, h, w):
    cost1 = costs[0:w * (h-1)]
    cost1 = np.reshape(cost1, (h-1, w), 'C')
    cost2 = costs[w*(h-1):]
    cost2 = np.reshape(cost2, (h, w-1), 'C')
    return cost1, cost2


def save_results_npy(U, V, X_unw, X_nunw, X_wrapped, error, error_noisy, X_real, X_sigma_noise, duration=0, path="./temp_results/"):
    if not os.path.isdir(path):
        os.mkdir(path)
    
    np.save(path + "output.npy", U)
    np.save(path + "output_reconstructed_with_gradients.npy", V)
    np.save(path + "X_unw.npy", X_unw)
    np.save(path + "X_wrapped.npy", X_wrapped)
    np.save(path + "error.npy", error)
    np.save(path + "duration.npy", duration)

def get_simu_data(location, path="topo_phase_dataset_v1/"):

    full_path = path + location + "/" + "npy_files/"
    X_real = np.load(full_path + "real.npy")
    X_sigma_noise = np.load(full_path + "sigma_noise.npy")
    added_noise = X_sigma_noise * np.random.randn(*X_sigma_noise.shape)
    X_simu_noisy_wrapped = np.zeros_like(X_real)
    if os.path.isfile(full_path + "simu_noisy_wrapped.npy"):
        X_simu_noisy_wrapped = np.load(full_path + "simu_noisy_wrapped.npy")
    
    X_real_goldstein = X_real
    if os.path.isfile(full_path + "real_goldstein.npy"):
        X_real_goldstein = np.load(full_path + "real_goldstein.npy")

    X_simu_unw = np.load(full_path + "simu_unwrapped.npy")
    X_simu_noisy_unw = added_noise + X_simu_unw
    return X_real, X_real_goldstein, added_noise, X_simu_noisy_wrapped, X_simu_unw, X_simu_noisy_unw


def load_data(location, path="./topo_phase_data_v1/data_2048/", noise_level="noiseless"):
    X = 0
    X_unw = 0

    X_real, X_real_goldstein, added_noise, X_simu_noisy_wrapped, X_unw, X_nunw = get_simu_data(location, path=path)
    X_full = X_simu_noisy_wrapped
    # X_full = X_simu_unw
    if noise_level == "noise_non_uniform":
        X = X_full
    elif noise_level == "noiseless":
        X = wrap_matrix(X_unw)  # Assuming wrap_matrix is defined
    elif noise_level == "real":
        X = X_real
    elif noise_level == "real_goldstein":
        X = X_real_goldstein
        X_real = X_real_goldstein

    print(np.shape(X))
    return X, X_real, X_unw, X_nunw, added_noise

def load_weights(path):
    c1 = np.load(path + "/c1.npy")
    c2 = np.load(path + "/c2.npy")
    return c1, c2

def scale_result(U, original_image):
    alpha = np.sum(original_image - U) / (U.shape[0] * U.shape[1])
    return U + alpha

def compute_error(U, original_image):
    scaled_U = scale_result(U, original_image)
    return scaled_U - original_image


def reconstruct_image_from_gradients(SU, UT):
    N = UT.shape[0]
    M = SU.shape[1]

    SU_partial_sum = np.zeros(N-1)
    SU_partial_sum[0] = SU[0, 0]
    for i in range(1, N-1):
        SU_partial_sum[i] = SU_partial_sum[i-1] + SU[i, 0]

    UT_partial_sum = np.zeros((N, M-1))
    for i in range(N):
        UT_partial_sum[i, 0] = UT[i, 0]
        for j in range(1, M-1):
            UT_partial_sum[i, j] = UT_partial_sum[i, j-1] + UT[i, j]

    U = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            if i == 0 and j == 0:
                U[i, j] = 0
            elif i == 0:
                U[i, j] = UT_partial_sum[i, j-1]
            elif j == 0:
                U[i, j] = SU_partial_sum[i-1]
            else:
                U[i, j] = SU_partial_sum[i-1] + UT_partial_sum[i, j-1]

    return U