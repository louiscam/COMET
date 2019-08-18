import os
os.chdir('/Users/louis.cammarata/Documents/Harvard/Fall2018/Research/Data')

import xlmhg
import hgmd-v1 as hgmd
import GenerateSyntheticExpressionMatrix as gsec
import math  
import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import time
from tqdm import tqdm

os.chdir('/Users/louis.cammarata/Documents/Harvard/Fall2018/Research/COMETFinalDraft/COMETDraftFiguresv4/Normal')

# Likelihood Ratio Test for Logistic Regression

def LRT_LogReg(df):
    # Define model matrix and response
    X = np.matrix(df.drop('cluster', axis=1))
    y = df['cluster']
    # Train logistic regression with full model
    logreg1 = LogisticRegression().fit(X,y)
    ll1 = -log_loss(y,logreg1.predict_proba(X),normalize=False)
    # Train logistic regression with null model (only intercept)
    logreg0 = LogisticRegression().fit([[0]]*len(X) ,y)
    ll0 = -log_loss(y,logreg0.predict_proba(X),normalize=False)
    # Likelihood ratio test
    stat = 2*(ll1-ll0)
    pval = ss.chi2.sf(stat, 1)
    return(pval)
    


# # I. Fixed high sample size, Running Effect Size

# Set seed
np.random.seed(13)

# Define parameters
n_cells = 5000
cluster_size = 500
c1_var = 1
c0_var = 5
repeat = 100

# Define range of variation in mean difference and effect size
epsilon_range = np.arange(0,4,0.1)

# Initialize record vectors
xlmHG_pv, mHG_pv, t_pv, w_pv, ks_pv, lr_pv = [], [], [], [], [], []
xlmHG_pv_sd, mHG_pv_sd, t_pv_sd, w_pv_sd, ks_pv_sd, lr_pv_sd = [], [], [], [], [], []
# Generate expression matrices and populate memory arrays
for e in tqdm(epsilon_range):
    time.sleep( .01 )
    
    # Initialize mean values
    xlmHG, mHG, t, w, ks, lr = [], [], [], [], [], []
    
    for k in np.arange(0,repeat,1):
        # Generate gene expression matrix
        cell_data = gsec.createExpressionMat(n_cells=n_cells,
                                            cluster_size=cluster_size,
                                            c1_loc=e, c1_var=c1_var, 
                                            c0_loc=0,c0_var=c0_var,
                                            n_outliers=0,
                                            cout_loc=0,cout_var=0,
                                            normal_dist=True)
        # Perform tests
        test_results = hgmd.singleton_test(cell_data,1,start=1,X=np.int(0.15*cluster_size),L=np.int(2*cluster_size))        
        test_results0 = hgmd.singleton_test(cell_data,1,start=1) #XL1
        # Fill in vectors
        xlmHG.append(float(test_results['mHG_pval']))
        mHG.append(float(test_results0['mHG_pval']))
        t.append(float(test_results['t_pval']))
        w.append(float(test_results['w_pval']))
        ks.append(float(test_results['ks_pval']))
        lr.append(float(LRT_LogReg(cell_data)))
    
    # Keep track of values
    xlmHG_pv.append(np.mean(xlmHG))
    xlmHG_pv_sd.append(np.var(xlmHG)**0.5)
    mHG_pv.append(np.mean(mHG))
    mHG_pv_sd.append(np.var(mHG)**0.5)
    t_pv.append(np.mean(t))
    t_pv_sd.append(np.var(t)**0.5)
    w_pv.append(np.mean(w))
    w_pv_sd.append(np.var(w)**0.5)
    ks_pv.append(np.mean(ks))
    ks_pv_sd.append(np.var(ks)**0.5)
    lr_pv.append(np.mean(lr))
    lr_pv_sd.append(np.var(lr)**0.5)

# Plot without x axis break

# Plot p-value vs. True Effect Size Across Clusters for the 3 different tests
plt.errorbar(epsilon_range,np.array(t_pv),yerr = np.array(t_pv_sd),color='blue',marker='D', linestyle = 'None')
plt.errorbar(epsilon_range,w_pv,yerr = np.array(w_pv_sd),color='green',marker='s', linestyle = 'None')
plt.errorbar(epsilon_range,ks_pv,yerr = np.array(ks_pv_sd),color='orange',marker='P', linestyle = 'None')
plt.errorbar(epsilon_range,lr_pv,yerr = np.array(lr_pv_sd),color='magenta',marker='*', linestyle = 'None')
plt.errorbar(epsilon_range,np.array(mHG_pv),yerr = np.array(mHG_pv_sd),color='red',marker='X', linestyle = 'None')
plt.plot([0, np.max(epsilon_range)], [0.05, 0.05], 'k-', lw=0.5)
#plt.title('p-Value vs. Mean Dispersion ('+str(n_cells)+' cells)')
plt.xlabel('Difference in mean between clusters')
plt.ylabel('p-Value')
plt.xscale('log')
plt.legend( [ "0.05", "t", "WRS","KS", "LRT (LR)","XL-mHG" ],bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
#plt.savefig('pvalMeanDiff-Normal.eps', format='eps', dpi=1000,bbox_inches='tight')
plt.show()

# Figure with broken axes
#f,(ax1,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')

m1, M1 = -0.05,0.75
m2, M2 = 2.75,4

prop = (M1-m1)/(M1+M2-m1-m2)
f = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[prop, 1-prop]) 


# Plot data on ax1
ax1 = plt.subplot(gs[0])
ax1.errorbar(epsilon_range,np.array(t_pv),yerr = [np.minimum(np.array(t_pv_sd),np.array(t_pv)),np.array(t_pv_sd)],color='blue',marker='D', linestyle = 'None')
ax1.errorbar(epsilon_range,w_pv,yerr = np.array(w_pv_sd),color='green',marker='s', linestyle = 'None')
ax1.errorbar(epsilon_range,np.array(ks_pv)+0.02,yerr = np.array(ks_pv_sd),color='orange',marker='P', linestyle = 'None')
ax1.errorbar(epsilon_range,lr_pv,yerr = np.array(lr_pv_sd),color='magenta',marker='*', linestyle = 'None')
ax1.errorbar(epsilon_range,np.array(mHG_pv)+0.01,yerr = [np.array(mHG_pv_sd),np.minimum(np.array(mHG_pv_sd),1-np.array(mHG_pv))],color='brown',marker='X', linestyle = 'None')
ax1.errorbar(epsilon_range,np.array(xlmHG_pv),yerr = [np.array(xlmHG_pv_sd),np.minimum(np.array(xlmHG_pv_sd),1-np.array(xlmHG_pv))],color='red',marker='X', linestyle = 'None')
ax1.plot([0, np.max(epsilon_range)], [0.05, 0.05], 'k-', lw=0.5)
# Plot data on ax2
ax2 = plt.subplot(gs[1])
ax2.errorbar(epsilon_range,np.array(t_pv),yerr = np.array(t_pv_sd),color='blue',marker='D', linestyle = 'None')
ax2.errorbar(epsilon_range,w_pv,yerr = np.array(w_pv_sd),color='green',marker='s', linestyle = 'None')
ax2.errorbar(epsilon_range,np.array(ks_pv)+0.02,yerr = np.array(ks_pv_sd),color='orange',marker='P', linestyle = 'None')
ax2.errorbar(epsilon_range,lr_pv,yerr = np.array(lr_pv_sd),color='magenta',marker='*', linestyle = 'None')
#ax2.errorbar(epsilon_range,np.array(mHG_pv),yerr = [np.array(mHG_pv_sd),np.minimum(np.array(mHG_pv_sd),1-np.array(mHG_pv))],color='red',marker='X', linestyle = 'None')
ax2.errorbar(epsilon_range,np.array(mHG_pv)+0.01,yerr = [np.minimum(np.array(mHG_pv_sd),np.array(mHG_pv)),np.minimum(np.array(mHG_pv_sd),1-np.array(mHG_pv))],color='brown',marker='X', linestyle = 'None')
ax2.errorbar(epsilon_range,np.array(xlmHG_pv),yerr = [np.minimum(np.array(xlmHG_pv_sd),np.array(xlmHG_pv)),np.minimum(np.array(xlmHG_pv_sd),1-np.array(xlmHG_pv))],color='red',marker='X', linestyle = 'None')
ax2.plot([0, np.max(epsilon_range)], [0.05, 0.05], 'k-', lw=0.5)

ax1.set_xlim(m1,M1)
ax1.set_ylim(-0.05,1.05)
ax2.set_xlim(m2,M2)
ax2.set_ylim(-0.05,1.05)

# hide the spines between ax and ax2
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax1.yaxis.tick_left()
ax1.tick_params(labelright='off')
ax2.set_yticks([])

d = .015
'''kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1,1), (-d,+d), **kwargs, label='_nolegend_')
ax1.plot((1,1),(1-d,1+d), **kwargs,label='_nolegend_')
kwargs.update(transform=ax2.transAxes)
ax2.plot((0,0), (1-d,1+d), **kwargs,label='_nolegend_')
ax2.plot((0,0), (-d,+d), **kwargs,label='_nolegend_')'''
plt.legend( [ "0.05", "t-test", 
             "Wilcoxon Rank Sum test","Kolmogorov-Smirnov", 
             "Likelihood Ratio Test (LR)","mHG test", 'XL-mHG test' ],bbox_to_anchor=(1.1,0.5), loc="center left", borderaxespad=0, fontsize = 14)
plt.xlabel('Mean Difference', fontsize = 20, x = 0.1, y = 1)
ax1.set_ylabel('p-Value', fontsize = 20)
#plt.savefig('pvalMeanDiff-Normal-Break.png', format='png', dpi=1000,bbox_inches='tight')
plt.show()


# # II. Running Sample Size, Fixed Small Mean Difference

# Set seed
np.random.seed(13)

# Define parameters
repeat = 100
e=1
c1_var = 1
c0_var = 5

# Initialize record vectors
xlmHG_pv, mHG_pv, t_pv, w_pv, ks_pv, lr_pv = [], [], [], [], [], []
xlmHG_pv_sd, mHG_pv_sd, t_pv_sd, w_pv_sd, ks_pv_sd, lr_pv_sd = [], [], [], [], [], []

# Define range of variation in sample size
#nrange = np.arange(100,10000,200)
nrange = np.logspace(1,4,10,dtype='int')

# Generate expression matrices and populate memory arrays
for n in tqdm(nrange):
    time.sleep( .01 )
    
    # Initialize mean values
    xlmHG, mHG, t, w, ks, lr = [], [], [], [], [], []
    
    # Define cluster size and compute pooled standard deviations
    cluster_size = np.int(0.1*n)
    
    for k in np.arange(0,repeat):
        # Generate gene expression matrix
        cell_data = gsec.createExpressionMat(n_cells=n,
                                            cluster_size=cluster_size,
                                            c1_loc=e, c1_var=c1_var, 
                                            c0_loc=0,c0_var=c0_var,
                                            n_outliers=0,
                                            cout_loc=0,cout_var=0,
                                            normal_dist=True)

        # Perform tests
        test_results = hgmd.singleton_test(cell_data,1,start=1,X=np.int(0.15*cluster_size),L=np.int(2*cluster_size))        
        test_results0 = hgmd.singleton_test(cell_data,1,start=1) #XL1
        # Fill in vectors
        xlmHG.append(float(test_results['mHG_pval']))
        mHG.append(float(test_results0['mHG_pval']))
        t.append(float(test_results['t_pval']))
        w.append(float(test_results['w_pval']))
        ks.append(float(test_results['ks_pval']))
        lr.append(float(LRT_LogReg(cell_data)))
    
    # Keep track of values
    xlmHG_pv.append(np.mean(xlmHG))
    xlmHG_pv_sd.append(np.var(xlmHG)**0.5)
    mHG_pv.append(np.mean(mHG))
    mHG_pv_sd.append(np.var(mHG)**0.5)
    t_pv.append(np.mean(t))
    t_pv_sd.append(np.var(t)**0.5)
    w_pv.append(np.mean(w))
    w_pv_sd.append(np.var(w)**0.5)
    ks_pv.append(np.mean(ks))
    ks_pv_sd.append(np.var(ks)**0.5)
    lr_pv.append(np.mean(lr))
    lr_pv_sd.append(np.var(lr)**0.5)
    


# # Plot p-value vs. Sample Size for the 3 different tests

# Plot p-value vs. Sample Size for the 3 different tests¶
f = plt.figure(figsize=(8, 6)) 
plt.plot([np.min(nrange), np.max(nrange)], [0.05, 0.05], 'k-', lw=0.5)
plt.errorbar(nrange,np.array(t_pv),yerr = [np.minimum(np.array(t_pv_sd),np.array(t_pv)),np.array(t_pv_sd)],color='blue',marker='D', linestyle = 'None')
plt.errorbar(nrange,w_pv,yerr = [np.minimum(np.array(w_pv_sd),np.array(w_pv)),np.array(w_pv_sd)],color='green',marker='s', linestyle = 'None')
plt.errorbar(nrange,ks_pv,yerr = np.array(ks_pv_sd),color='orange',marker='P', linestyle = 'None')
plt.errorbar(nrange,lr_pv,yerr = np.array(lr_pv_sd),color='magenta',marker='*', linestyle = 'None')
plt.errorbar(nrange,np.array(mHG_pv),yerr = [np.array(mHG_pv_sd),np.minimum(np.array(mHG_pv_sd),1-np.array(mHG_pv))],color='brown',marker='X', linestyle = 'None')
plt.errorbar(nrange,np.array(xlmHG_pv),yerr = [np.array(xlmHG_pv_sd),np.minimum(np.array(xlmHG_pv_sd),1-np.array(xlmHG_pv))],color='red',marker='X', linestyle = 'None')

#plt.title('p-Value vs. Sample Size (mean diff = '+str(e)+')')
plt.xlabel('Sample Size', fontsize = 20)
plt.ylabel('p-Value', fontsize = 20)
plt.legend( [ "0.05", "t-test", 
             "Wilcoxon Rank Sum test","Kolmogorov-Smirnov", 
             "Likelihood Ratio Test (LR)","mHG test", "XL-mHG test" ], bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, fontsize = 14)
plt.xscale('log')
plt.ylim(-0.05,1.05)
#plt.xlim([np.min(nrange)-10, np.max(nrange)])
#plt.savefig('pvalSampleSize-Normal.png', format='png', dpi=1000,bbox_inches='tight')
plt.show()



