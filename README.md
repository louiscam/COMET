# COMET

## Description

This repository contains scripts for the simulations implemented in the COMET article by Delaney et al. (2019). The pre-print can be found at https://www.biorxiv.org/content/10.1101/655753v1. Documentation on the COMET Python package is available at https://hgmd.readthedocs.io/en/latest/index.html. Simulation figures in the COMET article can be reproduced using the scripts below.

* hgmd-v1.py and hgmd-v2.py are modified versions of the COMET script (https://github.com/Cnrdelaney/HG_marker_detection.git) for simulation purposes.
* GenerateSyntheticExpressionMatrix.py generates Gaussian expression values for one gene in many cells. This script is used in Simulations-TestComparisons-Normal.py and Simulations-TestComparisons-NegBin.py.
* Simulations-TestComparisons-Normal.py and Simulations-TestComparisons-NegBin.py compare COMET to standard statistical tests used in gene differential expression testing.
* Simulations-ClassifierComparisons-Gaussian.py and Simulations-ClassifierComparisons-PoissonGamma.py compare COMET to standard classifiers including logistic regression and tree ensembles 	methods.

Please refer to the COMET article for further details on the design and implementation of the above simulation scripts.

## Author

Louis Cammarata