import os
os.chdir('/Users/louis.cammarata/Documents/Harvard/Fall2018/Research/Data')

import hgmd_code as hgmd
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib import cm

# Generate cells-by-gene matrix

def createExpressionMat(n_cells,
                        cluster_size,
                        c1_loc, c1_var, 
                        c0_loc,c0_var,
                        n_outliers,
                        cout_loc,cout_var,
                        normal_dist=True):
    
    # Create cell names
    index = ['cell_'+str(i) for i in np.arange(0,n_cells)]
        
    # Create cluster assignment vector
    cluster = np.concatenate((np.repeat(1,cluster_size),np.repeat(0,n_cells-cluster_size)),axis=0)
    
    # Create gene expression vector
    if (normal_dist==True):
        # Create gene expression vector in cluster 0
        gene_c0 = np.random.normal(loc=c0_loc,scale=c0_var,size=n_cells-cluster_size)
        # Create gene expression vector in cluster 1 for non outliers
        gene_c1_std = np.random.normal(loc=c1_loc,scale=c1_var,size=cluster_size-n_outliers)
        # Create gene expression vector in cluster 1 for outliers
        gene_c1_out = np.random.normal(loc=cout_loc,scale=cout_var,size=n_outliers)
    else:
        # Create gene expression vector in cluster 0
        gene_c0 = ss.cauchy.rvs(loc=c0_loc,scale=c0_var,size=n_cells-cluster_size)
        # Create gene expression vector in cluster 1 for non outliers
        gene_c1_std = ss.cauchy.rvs(loc=c1_loc,scale=c1_var,size=cluster_size-n_outliers)
        # Create gene expression vector in cluster 1 for outliers
        gene_c1_out = ss.cauchy.rvs(loc=cout_loc,scale=cout_var,size=n_outliers)
    gene_exp = np.concatenate((gene_c1_out,gene_c1_std,gene_c0),axis=0)
    
    # Create dataframe
    tmp = {'cluster':cluster,'gene':gene_exp}
    cell_data = pd.DataFrame(tmp,index=index)
    
    return(cell_data)
    
def createNBExpressionMat(n_cells,
                        cluster_size,
                        c1_n, c1_p, 
                        c0_n,c0_p,
                        n_outliers,
                        cout_n,cout_p):
    
    # Create cell names
    index = ['cell_'+str(i) for i in np.arange(0,n_cells)]
        
    # Create cluster assignment vector
    cluster = np.concatenate((np.repeat(1,cluster_size),np.repeat(0,n_cells-cluster_size)),axis=0)
    
    # Create gene expression vector
    # Create gene expression vector in cluster 0
    gene_c0 = np.random.negative_binomial(n=c0_n, p=c0_p, size=n_cells-cluster_size)
    # Create gene expression vector in cluster 1 for non outliers
    gene_c1_std = np.random.negative_binomial(n=c1_n, p=c1_p, size=cluster_size-n_outliers)
    # Create gene expression vector in cluster 1 for outliers
    gene_c1_out = np.random.negative_binomial(n=cout_n, p=cout_p, size=n_outliers)
    gene_exp = np.concatenate((gene_c1_out,gene_c1_std,gene_c0),axis=0)
    # Normalize
    gene_exp = gene_exp*np.power(10,6)/np.sum(gene_exp)
    
    # Create dataframe
    tmp = {'cluster':cluster,'gene':gene_exp}
    cell_data = pd.DataFrame(tmp,index=index)
    
    return(cell_data)

def createNBExpressionMat2(n_cells,
                        cluster_size,
                        c1_n, c1_p, 
                        c0_n,c0_p,
                        offset,
                        n_outliers,
                        cout_n,cout_p):
    
    # Create cell names
    index = ['cell_'+str(i) for i in np.arange(0,n_cells)]
        
    # Create cluster assignment vector
    cluster = np.concatenate((np.repeat(1,cluster_size),np.repeat(0,n_cells-cluster_size)),axis=0)
    
    # Create gene expression vector
    # Create gene expression vector in cluster 0
    gene_c0 = np.random.negative_binomial(n=c0_n, p=c0_p, size=n_cells-cluster_size)
    # Create gene expression vector in cluster 1 for non outliers
    gene_c1_std = np.random.negative_binomial(n=c1_n, p=c1_p, size=cluster_size-n_outliers)+offset
    # Create gene expression vector in cluster 1 for outliers
    gene_c1_out = np.random.negative_binomial(n=cout_n, p=cout_p, size=n_outliers)
    gene_exp = np.concatenate((gene_c1_out,gene_c1_std,gene_c0),axis=0)
    # Normalize
    #gene_exp = gene_exp*np.power(10,6)/np.sum(gene_exp)
        
    # Create dataframe
    tmp = {'cluster':cluster,'gene':gene_exp}
    cell_data = pd.DataFrame(tmp,index=index)
    
    return(cell_data)
    
# Generate cells-by-genes matrix

def createMultipleExpressionMat(n_cells,
                                cluster_size,
                                params, 
                                c0_loc,c0_var,
                                normal_dist=True):
    
    # Create cell names
    index = ['cell_'+str(i) for i in np.arange(0,n_cells)]
    # Create gene names
    n_genes = len(params)
    genes = ['gene_'+str(i) for i in np.arange(0,n_genes)]
        
    # Create cluster assignment vector
    cluster = np.concatenate((np.repeat(1,cluster_size),np.repeat(0,n_cells-cluster_size)),axis=0)

    # Create dataframe
    tmp = {'cluster':cluster}
    cell_data = pd.DataFrame(tmp,index=index)
    
    for g in np.arange(0,n_genes):
        # Create gene expression vector
        if (normal_dist==True):
            # Create gene expression vector in cluster 0
            gene_c0 = np.random.normal(loc=c0_loc,scale=c0_var,size=n_cells-cluster_size)
            # Create gene expression vector in cluster 1 for non outliers
            gene_c1 = np.random.normal(loc=params[g][0],scale=params[g][1],size=cluster_size)
        else:
            # Create gene expression vector in cluster 0
            gene_c0 = ss.cauchy.rvs(loc=c0_loc,scale=c0_var,size=n_cells-cluster_size)
            # Create gene expression vector in cluster 1 for non outliers
            gene_c1 = ss.cauchy.rvs(loc=params[g][0],scale=params[g][1],size=cluster_size)
        gene_exp = np.concatenate((gene_c1,gene_c0),axis=0)
        cell_data[genes[g]] = gene_exp

    return(cell_data)
    
