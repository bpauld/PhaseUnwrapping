# Phase Unwrapping via Iteratively Reweighted Least Squares

This repository contains code from our paper on 2D phase unwrapping using iteratively reweighted least squares, available [here](https://arxiv.org/abs/2401.09961).

## Installation

Once you have downloaded the repository, we suggest creating a `conda` environment named `phase_unwrapping`.
```
conda env create -f environment.yml
conda activate phase_unwrapping
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

## Acknowledgements

To unwrap the image, you can use weights generated from the SNAPHU software. SNAPHU does not offer the option to only compute weights, so we slightly modifiy the original code and distribute it here. The copyright notice for the SNAPHU software is as follows:

        Copyright
        ---------

        Copyright 2002 Board of Trustees, Leland Stanford Jr. University

        Except as noted below, permission to use, copy, modify, and
        distribute, this software and its documentation for any purpose is
        hereby granted without fee, provided that the above copyright notice
        appear in all copies and that both that copyright notice and this
        permission notice appear in supporting documentation, and that the
        name of the copyright holders be used in advertising or publicity
        pertaining to distribution of the software with specific, written
        prior permission, and that no fee is charged for further distribution
        of this software, or any modifications thereof.  The copyright holder
        makes no representations about the suitability of this software for
        any purpose.  It is provided "as is" without express or implied
        warranty.

        THE COPYRIGHT HOLDER DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
        SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
        FITNESS, IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
        SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
        RESULTING FROM LOSS OF USE, DATA, PROFITS, QPA OR GPA, WHETHER IN AN
        ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
        OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

        The parts of this software derived from the CS2 minimum cost flow
        solver written by A. V. Goldberg and B. Cherkassky are governed by the
        terms of the copyright holder of that software.  Permission has been
        granted to use and distrubute that software for strictly noncommercial
        purposes as part of this package, provided that the following
        copyright notice from the original distribution and URL accompany the
        software:

        COPYRIGHT C 1995 IG Systems, Inc.  Permission to use for
        evaluation purposes is granted provided that proper
        acknowledgments are given.  For a commercial licence, contact
        igsys@eclipse.net.

        This software comes with NO WARRANTY, expressed or implied. By way
        of example, but not limitation, we make no representations of
        warranties of merchantability or fitness for any particular
        purpose or that the use of the software components or
        documentation will not infringe any patents, copyrights,
        trademarks, or other rights.

## Citing our work

To cite our our work please use:
```
@article{dubois2024iteratively,
  title={Iteratively Reweighted Least Squares for Phase Unwrapping},
  author={Dubois-Taine, Benjamin and Akiki, Roland and d'Aspremont, Alexandre},
  journal={arXiv preprint arXiv:2401.09961},
  year={2024}
}
```
