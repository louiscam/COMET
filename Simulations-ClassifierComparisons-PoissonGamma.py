import os
os.chdir('/Users/louis.cammarata/Documents/Harvard/Fall2018/Research/Data')
#os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Standard packages
import xlmhg
import hgmd-v1 as hgmd
import hgmd-v2 as new
import GenerateSyntheticExpressionMatrix as gsec
import math  
import pandas as pd
import numpy as np
import scipy.stats as ss
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from tqdm import tqdm
import random
import hgmd as comet
from brokenaxes import brokenaxes

# Classifiers
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

os.chdir('/Users/louis.cammarata/Documents/Harvard/Fall2018/Research/COMETFinalDraft/COMETDraftFiguresv4')


# # Generate Expression Matrix

# Arguments
# p = numbr of genes, p1 = number of good markers, p2 = number of poor markers, p3 = number of non markers
# n = number of cells, n1 = number of cells in cluster C1, n0 = number of cells in cluster C0
# beta = shape parameter of the Gamma distribution for gene mean
# alpha_back = rate parameter for non markers, alpha_1 = rate parameter for good markers, alpha_2 = for bad markers
# q = proportion of C1 cells in which poor markers are expressed
# eff = efficiency noise parameter
# cpm = should the data be normalized to CPM, log = should the data be log-transformed with 1 pseudocount

def simulateExpMatrix(p,p1,p2,p3,
                      n, n1, n0,
                      beta, alpha_back, alpha_1, alpha_2,
                      q, eff,
                      cpm, log):

    # Initialize expression matrix (include technical zeros)
    X = np.empty((p,n))
    # Background
    for g in np.arange(p):
        mu = np.random.gamma(shape = alpha_back, scale = beta, size = 1)
        X[g,] = np.random.poisson(lam = mu, size = n)
    # Good markers
    for g in np.arange(p1):
        mu = np.random.gamma(shape = alpha_1, scale = beta, size = 1)
        X[g,0:n1] = np.random.poisson(lam = mu, size = n1)
    # Poor markers
    for g in np.arange(p1,p1+p2,1):
        mu = np.random.gamma(shape = alpha_2, scale = beta, size = 1)
        ncells = int(q*n1)
        cells = np.random.choice(np.arange(n1),ncells, replace=False)
        X[g,cells] = np.random.poisson(lam = mu, size = ncells) 
    X_truth = np.array(X)
    # Reproduce efficiency noise
    scale_factor = np.random.uniform(low = 1-eff, high = 1+eff, size =(1,n))
    
    X_noisy = np.rint(np.random.poisson(lam = X_truth)*scale_factor)
    X_noisy = np.matrix(X_noisy)
    
    ''''X_noisy = X_truth*scale_factor
    # Reproduce technical noise
    X_noisy = np.random.poisson(lam = X_noisy)
    X_noisy = np.matrix(X_noisy) '''  
    
    
    # Normalize to CPM and log-transform
    if cpm==True:
        X_noisy = X_noisy/np.sum(X_noisy,0)*1e4
    if log==True:
        X_noisy = np.log2(1+X_noisy)
    # Create dataframe
    df = pd.DataFrame(X_noisy,dtype='float64')
    df.columns = ['cell_'+str(i) for i in np.arange(n)]
    df.index = ['gene_'+str(i) for i in np.arange(p)]
    df = np.transpose(df)
    df['cluster'] = np.concatenate((np.repeat(1,n1),np.repeat(0,n0)))
    
    return(df)

# Number of genes
p = 1000
p1 = 50; p2 = 50; p3 = 950
# Number of cells
n = 500
n1 = 50; n0 = n-n1
# Gamma Parameters
beta = 1
alpha_back = 2**4; alpha_1 =  2**5; alpha_2 = 2**8
# Expression of poor markers in q% of cells
q = 0.1
# Efficiency noise
eff = 0.05

df = simulateExpMatrix(p,p1,p2,p3,
                      n, n1, n0,
                      beta, alpha_back, alpha_1, alpha_2,
                      q, eff,
                      cpm = False, log = False)

X = np.matrix(df.drop('cluster', axis=1))[:3*n1,0:2*(p1+p2)]
f = plt.figure(figsize=(10, 5)) 
plt.imshow(X)
plt.xlabel('Genes', fontsize = 20)
plt.ylabel('Cells', fontsize = 20)
plt.xticks([],[])
plt.yticks([],[])
plt.colorbar(orientation = 'vertical', fraction = 0.015)
#plt.savefig('PoissonGammaMatrix.png', format='png', dpi=1000, bbox_inches = 'tight')
plt.show()


# # Marker detection with XL-mHG

def compute_mHG_SOR(df):
    # Define parameters
    n_genes = df.shape[1]
    n_cells = df.shape[0]
    n_cells1 = np.sum(np.array(df['cluster']))
    n_goodmark = int(0.05*(df.shape[1]-1))
    # Perform XL-mHG test
    xlmhg = new.batch_xlmhg(df.iloc[:,:n_genes], 
                            c_list=df['cluster'], 
                            coi=1, X=int(0.15*n_cells1),
                            L=min(n_cells,np.int(2*n_cells1)))
    # Sort by p-value
    ordered_genes = np.array(xlmhg.sort_values(by=['mHG_pval']).index)
    ranks_goodmark = np.where(np.in1d(ordered_genes,np.arange(n_goodmark)))[0]
    # Compute SOR
    SOR_mhg = np.sum(ranks_goodmark)/np.sum(np.arange(n_goodmark))
    return(SOR_mhg)


# # Marker detection with Extra Trees Classifier

# Define function

def compute_XT_SOR(df):
    # Define model matrix and response
    X = np.matrix(df.drop('cluster', axis=1))
    y = df['cluster']
    n_goodmark = int(0.05*(df.shape[1]-1))
    # Train random forest
    forest = ExtraTreesClassifier(n_estimators=250,
                                  criterion = 'gini',
                                  max_features = 'sqrt',
                                  bootstrap = True,
                                  oob_score = True,
                                  random_state=0).fit(X, y)    
    # Sort by feature importance
    ordered_genes = np.argsort(-forest.feature_importances_)
    # Compute SOR
    ranks_goodmark = np.where(np.in1d(ordered_genes,np.arange(n_goodmark)))[0]
    SOR_rf = np.sum(ranks_goodmark)/np.sum(np.arange(n_goodmark))   
    return([SOR_rf,forest.oob_score_])
    


# # Marker detection with Random Forest

# Define function

def compute_RF_SOR(df):
    # Define model matrix and response
    X = np.matrix(df.drop('cluster', axis=1))
    y = df['cluster']
    n_goodmark = int(0.05*(df.shape[1]-1))
    # Train random forest
    forest = RandomForestClassifier(n_estimators=250,
                                  criterion = 'gini',
                                  max_features = 'sqrt',
                                  bootstrap = True,
                                  oob_score = True,
                                  random_state=0).fit(X, y)    
    # Sort by feature importance
    ordered_genes = np.argsort(-forest.feature_importances_)
    # Compute SOR
    ranks_goodmark = np.where(np.in1d(ordered_genes,np.arange(n_goodmark)))[0]
    SOR_rf = np.sum(ranks_goodmark)/np.sum(np.arange(n_goodmark))   
    return([SOR_rf,forest.oob_score_])
    


# # Marker detection with Logistic Regression

# Likelihood Ratio Test for Logistic Regression

def LRT_LogReg(X,y):
    # Train logistic regression with full model
    logreg1 = LogisticRegression().fit(X,y)
    ll1 = -log_loss(y,logreg1.predict_proba(X),normalize=False)
    # Train logistic regression with null model (only intercept)
    logreg0 = LogisticRegression().fit([[0]]*len(X) ,y)
    ll0 = -log_loss(y,logreg0.predict_proba(X),normalize=False)
    # Likelihood ratio test
    stat = 2*(ll1-ll0)
    #pval = ss.chi2.sf(stat, 1)
    return(stat)
    
def compute_LR_SOR(df):
    # Define model matrix and response
    X = np.matrix(df.drop('cluster', axis=1))
    X = scale(X, axis=1, with_mean=True, with_std=True, copy=True)
    #X = np.matrix(df.drop('cluster', axis=1))>0
    #X = X.astype(int)
    y = df['cluster']
    n_goodmark = int(0.05*(df.shape[1]-1))
    # Train logistic regression
    LRT_pval = []
    for g in np.arange(df.shape[1]-1):
        pval = LRT_LogReg(np.matrix(X[:,g]).reshape((-1,1)),y)
        LRT_pval.append(pval)
    # Sort by measure of gene importance: logodds
    ordered_genes = np.argsort(-np.array(LRT_pval))
    # Compute SOR
    ranks_goodmark = np.where(np.in1d(ordered_genes,np.arange(n_goodmark)))[0]
    SOR_lr = np.sum(ranks_goodmark)/np.sum(np.arange(n_goodmark))      
    return(SOR_lr)


# # Analyze performance of XL-mHG vs. Random Forest 1

# Set seed
np.random.seed(13)

# Number of genes
p = 1000
p1 = int(0.05*p)
# Range of proportion of bad markers
pb_range = np.arange(0,0.18,0.03)
# Number of cells
n = 500
n1 = int(0.1*n); n0 = n-n1
# Gamma Parameters
beta = 0.1
alpha_back = 2**4; alpha_1 =  2**5; alpha_2 = 2**10
# Expression of poor markers in q% of cells
q = 0.1
# Efficiency noise
eff = 0.2
# Data transformation
cpm = False
log = False
# Number of runs to average over
repeat = 20

# Initialize record vectors
SOR_mhg, SOR_rf, oob_rf, SOR_xt, oob_xt, SOR_lr = [], [], [], [], [], []
sd_mhg, sd_rf, sd_oob_rf, sd_xt, sd_oob_xt, sd_lr = [], [], [], [], [], []

for pb in tqdm(pb_range):
    time.sleep( .01 )
    
    # Define number of bad markers
    p2 = int(pb*p)
    p3 = p-p1-p2
    
    # Initialize counters
    tmp_mhg, tmp_rf, tmp_oob_rf, tmp_xt, tmp_oob_xt, tmp_lr = [], [], [], [], [], []

    
    for i in np.arange(repeat):
        # Simulate expression matrix
        df = simulateExpMatrix(p,p1,p2,p3,
                               n, n1, n0,
                               beta, alpha_back, alpha_1, alpha_2,
                               q, eff,
                               cpm, log)
        tmp_mhg.append(compute_mHG_SOR(df))
        rf = compute_RF_SOR(df); tmp_rf.append(rf[0]); tmp_oob_rf.append(rf[1])
        xt = compute_XT_SOR(df); tmp_xt.append(xt[0]); tmp_oob_xt.append(xt[1])
        tmp_lr.append(compute_LR_SOR(df))
        
    # Compute SORs and respective standard deviations
    SOR_mhg.append(np.mean(tmp_mhg))
    SOR_rf.append(np.mean(tmp_rf))
    oob_rf.append(np.mean(tmp_oob_rf))
    SOR_xt.append(np.mean(tmp_xt))
    oob_xt.append(np.mean(tmp_oob_xt))
    SOR_lr.append(np.mean(tmp_lr))
    sd_mhg.append(np.var(tmp_mhg)**0.5)
    sd_rf.append(np.var(tmp_rf)**0.5)
    sd_oob_rf.append(np.var(tmp_oob_rf)**0.5)
    sd_xt.append(np.var(tmp_xt)**0.5)
    sd_oob_xt.append(np.var(tmp_oob_xt)**0.5)
    sd_lr.append(np.var(tmp_lr)**0.5)

# Plot results
plt.errorbar(pb_range,SOR_mhg,yerr = sd_mhg,color='blue',fmt = 'o')
plt.errorbar(pb_range,SOR_rf,yerr = sd_rf,color='orange',fmt = 'o')
plt.errorbar(pb_range,SOR_xt,yerr = sd_xt,color='red',fmt = 'o')
plt.errorbar(pb_range,SOR_lr,yerr = sd_lr,color='green',fmt = 'o')
#plt.title('SOR vs. Poor markers proportion (expressed in '+str(int(100*p))+'% C1 cells)')
plt.xlabel('Proportion of poor markers', fontsize = 20)
plt.ylabel('Scaled Sum of Ranks', fontsize = 20)
plt.legend(['XL-mHG','Random Forest', 'Extra Trees','Logistic regression'],
          bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, fontsize = 14)
#plt.savefig('SSRvsPropPoorMark-q10eff20.png', format='png', dpi=1000, bbox_inches = 'tight')
plt.show()

# Plot OOB
plt.errorbar(pb_range,1-np.array(oob_rf),yerr = sd_oob_rf,color='orange',fmt = 'o')
plt.errorbar(pb_range,1-np.array(oob_xt),yerr = sd_oob_xt,color='red',fmt = 'o')
plt.xlabel('Proportion of poor markers', fontsize = 20)
plt.ylabel('Out-of-Bag Error', fontsize = 20)
plt.legend(['Random Forest', 'Extra Trees'],loc = 'lower left', fontsize = 14)
#plt.savefig('OOBvsPropPoorMark-q10eff20.png', format='png', dpi=1000, bbox_inches = 'tight')
plt.show()


# # Analyze performance of XL-mHG vs. Random Forest 3

# Set seed
np.random.seed(13)

# Number of genes
p = 1000
p1 = int(0.05*p)
p2 = int(0.1*p)
# Number of cells
n = 500
n1 = int(0.1*n); n0 = n-n1
# Gamma Parameters
beta = 0.1
alpha_back = 2**4
alpha_1 = 2**5
range2 = 2**np.arange(5,15, 2)
# Expression of poor markers in q% of cells
q = 0.1
# Efficiency noise
eff = 0.2
# Data transformation
cpm = False
log = False
# Number of runs to average over
repeat = 20

# Initialize record vectors
SOR_mhg, SOR_rf, oob_rf, SOR_xt, oob_xt, SOR_lr = [], [], [], [], [], []
sd_mhg, sd_rf, sd_oob_rf, sd_xt, sd_oob_xt, sd_lr = [], [], [], [], [], []

for alpha_2 in tqdm(range2):
    time.sleep( .01 )
    
    # Initialize counters
    tmp_mhg, tmp_rf, tmp_oob_rf, tmp_xt, tmp_oob_xt, tmp_lr = [], [], [], [], [], []   
    
    for i in np.arange(repeat):
        # Simulate expression matrix
        df = simulateExpMatrix(p,p1,p2,p3,
                               n, n1, n0,
                               beta, alpha_back, alpha_1, alpha_2,
                               q, eff,
                               cpm, log)
        tmp_mhg.append(compute_mHG_SOR(df))
        rf = compute_RF_SOR(df); tmp_rf.append(rf[0]); tmp_oob_rf.append(rf[1])
        xt = compute_XT_SOR(df); tmp_xt.append(xt[0]); tmp_oob_xt.append(xt[1])
        tmp_lr.append(compute_LR_SOR(df))
        
    # Compute SORs and respective standard deviations
    SOR_mhg.append(np.mean(tmp_mhg))
    SOR_rf.append(np.mean(tmp_rf))
    oob_rf.append(np.mean(tmp_oob_rf))
    SOR_xt.append(np.mean(tmp_xt))
    oob_xt.append(np.mean(tmp_oob_xt))
    SOR_lr.append(np.mean(tmp_lr))
    sd_mhg.append(np.var(tmp_mhg)**0.5)
    sd_rf.append(np.var(tmp_rf)**0.5)
    sd_oob_rf.append(np.var(tmp_oob_rf)**0.5)
    sd_xt.append(np.var(tmp_xt)**0.5)
    sd_oob_xt.append(np.var(tmp_oob_xt)**0.5)
    sd_lr.append(np.var(tmp_lr)**0.5)

# Plot results
plt.errorbar(range2,SOR_mhg,yerr = sd_mhg,color='blue',fmt = 'o')
plt.errorbar(range2,SOR_rf,yerr = sd_rf,color='orange',fmt = 'o')
plt.errorbar(range2,SOR_xt,yerr = sd_xt,color='red',fmt = 'o')
plt.errorbar(range2,SOR_lr,yerr = sd_lr,color='green',fmt = 'o')
plt.xscale('log')
#plt.yscale('log')
#plt.title('SOR vs. Mean expression of poor markers (expressed in '+str(int(100*p))+'% C1 cells)')
plt.xlabel('Mean of Poor Markers', fontsize = 20)
plt.ylabel('Scaled Sum of Ranks', fontsize = 20)
plt.legend(['XL-mHG','Random Forest', 'Extra Trees','Logistic regression'],
          bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, fontsize = 14)
#plt.savefig('SSRvsMeanPoorMark-q10eff20.png', format='png', dpi=1000, bbox_inches = 'tight')
plt.show()

# Plot OOB
plt.errorbar(range2,1-np.array(oob_rf),yerr = sd_oob_rf,color='orange',fmt = 'o')
plt.errorbar(range2,1-np.array(oob_xt),yerr = sd_oob_xt,color='red',fmt = 'o')
plt.xscale('log')
plt.xlabel('Mean of poor markers', fontsize = 20)
plt.ylabel('Out-of-Bag Error', fontsize = 20)
plt.legend(['Random Forest', 'Extra Trees'],loc = 'upper right', fontsize = 14)
#plt.savefig('OOBvsMeanPoorMark-q10eff20.png', format='png', dpi=1000, bbox_inches = 'tight')
plt.show()

f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,8))

# plot the same data on both axes
ax.errorbar(range2,SOR_mhg,yerr = sd_mhg,color='blue',fmt = 'o')
ax.errorbar(range2,SOR_rf,yerr = sd_rf,color='orange',fmt = 'o')
ax.errorbar(range2,SOR_xt,yerr = sd_xt,color='red',fmt = 'o')
ax.errorbar(range2,SOR_lr,yerr = sd_lr,color='green',fmt = 'o')
ax2.errorbar(range2,SOR_mhg,yerr = sd_mhg,color='blue',fmt = 'o')
ax2.errorbar(range2,SOR_rf,yerr = sd_rf,color='orange',fmt = 'o')
ax2.errorbar(range2,SOR_xt,yerr = sd_xt,color='red',fmt = 'o')
ax2.errorbar(range2,SOR_lr,yerr = sd_lr,color='green',fmt = 'o')

# zoom-in / limit the view to different portions of the data
ax.set_ylim(15, 23)  # outliers only
ax2.set_ylim(0, 10)  # most of the data
ax.set_xscale('log')
ax2.set_xscale('log')

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax2.set_xlabel('Mean of Poor Markers', fontsize = 20)
ax2.set_ylabel('Scaled Sum of Ranks', fontsize = 20)
ax2.yaxis.set_label_coords(0.08, 0.45, transform=f.transFigure)

ax2.legend(['XL-mHG','Random Forest', 'Extra Trees','Logistic regression'],
          bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, fontsize = 14)

d = .015
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax2.transAxes) 
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs) 
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

#plt.savefig('SSRvsMeanPoorMark-q10eff2.png', format='png', dpi=1000, bbox_inches = 'tight')

plt.show()

