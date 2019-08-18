# COMET
This repository contains scripts for the simulations implemented in the COMET article by Delaney, C., Schnell, A., Cammarata, L., Yao-Smith, A., Regev, A., Kuchroo, V. K., \& Singer, M. (2019).

* hgmd-v1.py and hgmd-v2.py are modified versions of the XL-minimal HyperGeometric (XL-mHG) test script by Florian Wagner (see https://github.com/flo-compbio/xlmhg.git).\\
* GenerateSyntheticExpressionMatrix.py generates Gaussian expression values for one gene in many cells. This script is used in Simulations-TestComparisons-Normal.py and Simulations-TestComparisons-		NegBin.py.
* Simulations-TestComparisons-Normal.py and Simulations-TestComparisons-NegBin.py compare COMET to standard statistical tests used in gene differential expression testing.\\
* Simulations-ClassifierComparisons-Gaussian.py and Simulations-ClassifierComparisons-PoissonGamma.py compare COMET to standard classifiers including logistic regression and tree ensembles 	methods.
