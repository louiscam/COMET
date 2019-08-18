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

# Classifiers
from sklearn.preprocessing import scale
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier


# os.path.dirname(os.path.realpath(__file__))
os.chdir('/Users/louis.cammarata/Documents/Harvard/Fall2018/Research/COMETFinalDraft/COMETDraftFiguresv4')


# # Generate Expression Matrix

# Shuffle function (shuffles elements of each row in a matrix within a specific column range)

def shuffle(a,lim):
 
    # Partition a
    a1 = a[:,:lim]
    a2 = a[:,lim:]

    # Shuffle a1
    for i in np.arange(a1.shape[0]):
        np.random.shuffle(a1[i])
        
    # Reasemble a1 and a2
    a = np.concatenate((a1,a2),axis=1)
    return(a)

# Create generating function

def simulateExpMatrix(n_genes, n_goodmark, n_badmark,
                      n_cells, n_cells1,
                      e_good, e_bad,
                      p):
    
    # Additional parameters
    n_nomark = n_genes-n_goodmark-n_badmark
    n_cells0 = n_cells-n_cells1
    n_marked_bad = int(p*n_cells1)
    
    # Fill in good markers
    good_c1 = np.random.normal(e_good,1,(n_goodmark,n_cells1))
    good_c0 = np.random.normal(0,1,(n_goodmark,n_cells0))
    good = np.concatenate((good_c1,good_c0),axis=1)

    # Fill in bad markers
    bad_c1_1 = np.random.normal(e_bad,1,(n_badmark,n_marked_bad))
    bad_c1_2_c0 = np.random.normal(0,1,(n_badmark,n_cells-n_marked_bad))
    bad = np.concatenate((bad_c1_1,bad_c1_2_c0),axis=1)
    bad = shuffle(bad,n_cells1)

    # Fill in no markers
    no = np.random.normal(0,1,(n_nomark,n_cells))

    # Create dataframe
    combo_mat = np.concatenate((np.concatenate((good,bad),axis=0),no),axis=0)
    df = pd.DataFrame(combo_mat)
    df.columns = ['cell_'+str(i) for i in np.arange(n_cells)]
    df.index = ['gene_'+str(i) for i in np.arange(n_genes)]
    df = np.transpose(df)
    df['cluster'] = np.concatenate((np.repeat(1,n_cells1),np.repeat(0,n_cells0)))
    
    return(df)


# # Visualize expression matrix

# Set seed
np.random.seed(13)

# Genes Parameters
n_genes = 1000
n_goodmark = int(0.05*n_genes)
n_badmark = int(0.05*n_genes)
n_nomark = n_genes-n_goodmark-n_badmark

# Cells Parameters
n_cells = 200
n_cells1 = int(0.1*n_cells)
n_cells0 = n_cells-n_cells1
p = 0.1

# Offset for good and bad markers
e_good = 1
e_bad = 5

# Generate matrix
df = simulateExpMatrix(n_genes, n_goodmark, n_badmark,
                       n_cells, n_cells1,
                       e_good, e_bad,
                       p)

# Plot heatmap of model expression matrix
X = np.matrix(df.drop('cluster', axis=1))[:3*n_cells1,0:2*(n_goodmark+n_badmark)]
f = plt.figure(figsize=(10, 5)) 
plt.imshow(X)
plt.xlabel('Genes', fontsize = 20)
plt.ylabel('Cells', fontsize = 20)
plt.xticks([],[])
plt.yticks([],[])
plt.colorbar(orientation = 'vertical', fraction = 0.015)
#plt.savefig('Classifiers_ExpressionMatrix.eps', format='eps', dpi=1000, bbox_inches = 'tight')
plt.show()


# # Marker detection with XL-mHG

def compute_mHG_SOR(df):
    # Define parameters
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
    


# # Marker detection with Extra Trees Classifiers

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

def compute_LR_SOR_original(df):
    # Define model matrix and response
    X = np.matrix(df.drop('cluster', axis=1))
    X = scale(X, axis=1, with_mean=True, with_std=True, copy=True)
    #X = np.matrix(df.drop('cluster', axis=1))>0
    #X = X.astype(int)
    y = df['cluster']
    n_goodmark = int(0.05*(df.shape[1]-1))
    # Train logistic regression
    logodds = []
    for g in np.arange(df.shape[1]-1):
        logreg = LogisticRegression().fit(np.matrix(X[:,g]).reshape((-1,1)),y)
        logodds.append(logreg.coef_[0][0])
    # Sort by measure of gene importance: logodds
    ordered_genes = np.argsort(-np.array(logodds))
    # Compute SOR
    ranks_goodmark = np.where(np.in1d(ordered_genes,np.arange(n_goodmark)))[0]
    SOR_lr = np.sum(ranks_goodmark)/np.sum(np.arange(n_goodmark))      
    return(SOR_lr)
# # Analyze performance of XL-mHG vs. Random Forest 1

# Set seed
np.random.seed(13)

# Genes Parameters
n_genes = 1000
n_goodmark = int(0.05*n_genes)
# Cells Parameters
n_cells = 500
n_cells1 = int(0.1*n_cells)
# Offset for good and bad markers
e_good = 1
e_bad = 30
# Range of proportion of bad markers
pb_range = np.arange(0,0.11,0.01)
# Proportion of C1 cells expressing bad markers
p = 0.1
# Number of runs to average over
repeat = 20

# Initialize record vectors
SOR_mhg, SOR_rf, oob_rf, SOR_xt, oob_xt, SOR_lr = [], [], [], [], [], []
sd_mhg, sd_rf, sd_oob_rf, sd_xt, sd_oob_xt, sd_lr = [], [], [], [], [], []

for pb in tqdm(pb_range):
    time.sleep( .01 )
    
    # Define number of bad markers
    n_badmark = int(pb*n_genes)
    
    # Initialize counters
    tmp_mhg, tmp_rf, tmp_oob_rf, tmp_xt, tmp_oob_xt, tmp_lr = [], [], [], [], [], []
    
    for i in np.arange(repeat):
        # Simulate expression matrix
        df = simulateExpMatrix(n_genes, n_goodmark, n_badmark,
                               n_cells, n_cells1,
                               e_good, e_bad,
                               p)
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
plt.legend(['XL-mHG','Random Forest', 'Extra Trees','Logistic regression'],loc = 'upper left', fontsize = 14)
#plt.savefig('SSRvsPropPoorMark-MeanPoorMark30.eps', format='eps', dpi=1000, bbox_inches = 'tight')
plt.show()

# Plot OOB
plt.errorbar(pb_range,1-np.array(oob_rf),yerr = sd_oob_rf,color='orange',fmt = 'o')
plt.errorbar(pb_range,1-np.array(oob_xt),yerr = sd_oob_xt,color='red',fmt = 'o')
plt.xlabel('Proportion of poor markers', fontsize = 20)
plt.ylabel('Out-of-Bag Error', fontsize = 20)
plt.legend(['Random Forest', 'Extra Trees'],loc = 'lower left', fontsize = 14)
#plt.savefig('OOBvsPropPoorMark-MeanPoorMark30.png', format='png', dpi=1000, bbox_inches = 'tight')
plt.show()


# # Analyze performance of XL-mHG vs. Random Forest 3

# Set seed
np.random.seed(13)

# Genes Parameters
n_genes = 1000
n_goodmark = int(0.05*n_genes)
# Cells Parameters
n_cells = 500
n_cells1 = int(0.1*n_cells)
# Offset for good and bad markers
e_good = 1
e_bad_range = np.arange(0,33,3) #np.logspace(0,1.5,15)
# Proportion of bad marker
pb = 0.1
# Proportion of cells expressing bad markers
p = 0.1
# Number of runs to average over
repeat = 20

# Initialize record vectors
# Initialize record vectors
SOR_mhg, SOR_rf, oob_rf, SOR_xt, oob_xt, SOR_lr = [], [], [], [], [], []
sd_mhg, sd_rf, sd_oob_rf, sd_xt, sd_oob_xt, sd_lr = [], [], [], [], [], []

for e_bad in tqdm(e_bad_range):
    time.sleep( .01 )
    
    # Define number of poor markers
    n_badmark = int(pb*n_genes)
    
    # Initialize counters
    tmp_mhg, tmp_rf, tmp_oob_rf, tmp_xt, tmp_oob_xt, tmp_lr = [], [], [], [], [], []   
    for i in np.arange(repeat):
        # Simulate expression matrix
        df = simulateExpMatrix(n_genes, n_goodmark, n_badmark,
                               n_cells, n_cells1,
                               e_good, e_bad,
                               p)
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
plt.errorbar(e_bad_range,SOR_mhg,yerr = sd_mhg,color='blue',fmt = 'o')
plt.errorbar(e_bad_range,SOR_rf,yerr = sd_rf,color='orange',fmt = 'o')
plt.errorbar(e_bad_range,SOR_xt,yerr = sd_xt,color='red',fmt = 'o')
plt.errorbar(e_bad_range,SOR_lr,yerr = sd_lr,color='green',fmt = 'o')
#plt.xscale('log')
#plt.title('SOR vs. Mean expression of poor markers (expressed in '+str(int(100*p))+'% C1 cells)')
plt.xlabel('Mean of Poor Markers', fontsize = 20)
plt.ylabel('Scaled Sum of Ranks', fontsize = 20)
plt.legend(['XL-mHG','Random Forest', 'Extra Trees','Logistic Regression'],loc='upper left', fontsize = 14)
#plt.savefig('SSRvsMeanPoorMark.eps', format='eps', dpi=1000, bbox_inches = 'tight')
plt.show()

# Plot OOB
plt.errorbar(e_bad_range,1-np.array(oob_rf),yerr = sd_oob_rf,color='orange',fmt = 'o')
plt.errorbar(e_bad_range,1-np.array(oob_xt),yerr = sd_oob_xt,color='red',fmt = 'o')
plt.xlabel('Mean of poor markers', fontsize = 20)
plt.ylabel('Out-of-Bag Error', fontsize = 20)
plt.legend(['Random Forest', 'Extra Trees'],loc = 'upper left', fontsize = 14)
#plt.savefig('OOBvsMeanPoorMark.png', format='png', dpi=1000, bbox_inches = 'tight')
plt.show()


