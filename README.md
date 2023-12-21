# Phase Unwrapping via Iteratively Reweighted Least Squares

## Installation

Once you have downloaded the repository, we suggest creating a `conda` environment named `phase_unwrapping`.
```
conda env create -f environment.yml
conda activate phase_unwrap
```
We also provide a `requirements.txt` file for users wishing to use another virtual environment manager.

Once this is done, you need to compile SNAPHU. This step is not necessary if you plan to use your own weights or uniform weights in your objective function. However SNAPHU generated weights have shown good practical performance, so we encourage using them. In the root directory, run the following.
```
$ cd snaphu-v2.0.6
$ mkdir bin
$ cd src
$ make
```

## Running experiments

To reproduce the experiments, run in the working directory:

```
$ python final_script.py
```

This will load simulated and real images from a region in Lebanon from the `data` folder. It will then unwrap the images using the IRLS algorithm, and write the output to a `results` folder.

## Visualize results

To visualize the experiments, simply use the notebook `visualize_results.ipynb`.
For the experiments on real images, the plots should look like the following:

![Screenshot](screenshots/real_goldstein.png)

For the experiments on simulated images, the plots should look like the following:

![Screenshot](screenshots/noiseless.png)

## Citing our work

To cite our our work please use:
TODO
