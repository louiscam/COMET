# import re
import pandas as pd
import numpy as np
import xlmhg as hg
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages


#Single process, multiprocessed in parallel
def process(cluster,cell_data,X,L,min_exp_ratio,plot_pages,plot_genes,output_path):
    #print("Processing cluster " + str(cluster) + "...")
    cluster_path = output_path + "/cluster_" + str(cluster) + "/"
    os.makedirs(cluster_path, exist_ok=True)
    #print("Testing singletons...")
    singleton_data = singleton_test(cell_data, cluster, X, L)
    #print("Testing pairs...")
    pair_data = pair_test(
        cell_data, singleton_data, cluster, L, min_exp_ratio
    )
    #print("Calculating true positive/negative rates...")
    singleton_data, pair_data = find_TP_TN(
        cell_data, singleton_data, pair_data, cluster
    )
    #print("Calculating weighted TP/TN rates...")
    singleton_data, pair_data = find_weighted_TP_TN(
        cell_data, singleton_data, pair_data, cluster
    )
    #print("Saving to CSV...")
    singleton_data.to_csv(cluster_path + "singleton_data.csv")
    pair_data.to_csv(cluster_path + "pair_data.csv")
    #print("Done.")
    #print("Plotting true positive/negative rates...")
    make_TP_TN_plots(
        cell_data, singleton_data, pair_data, plot_genes,
        pair_path=(cluster_path + "TP_TN_plot.pdf"),
        singleton_path=(cluster_path + "singleton_TP_TN_plot.pdf")
    )
    #print("Done.")
    #print("Plotting discrete expression...")
    make_discrete_plots(
        cell_data, singleton_data, pair_data, plot_pages,
        path=(cluster_path + "discrete_plots.pdf"),
    )
    #print("Done.")
    #print("Plotting continuous expresssion...")
    make_combined_plots(
        cell_data, singleton_data, pair_data, plot_pages,
        pair_path=(cluster_path + "combined_plot.pdf"),
        singleton_path=(cluster_path + "singleton_combined_plot.pdf")
    )
    print("Done with cluster " + str(cluster))

    

def round_to_6(input_value):
    """Rounds a value to 6 significant figures."""

    from decimal import ROUND_HALF_EVEN, Context
    ctx = Context(prec=6, rounding=ROUND_HALF_EVEN)
    return float(ctx.create_decimal(input_value))


def fuzzy_rank(series):
    """Ranks a series of floats with 1e-6 relative fuzziness."""

    series = series.apply(round_to_6)
    return series.rank()


def get_discrete_exp(cells, singleton, start=3):
    """Creates discrete expression matrix using given cutoff."""

    exp = pd.DataFrame()
    for gene in cells.columns[start:]:
        cutoff = (
            singleton[
                singleton['gene'] == gene
            ]['mHG_cutoff_value'].iloc[0]
        )

        equal = pd.Series(np.isclose(cells[gene], cutoff))
        equal.index = cells.index
        less_than = cells[gene] < cutoff
        greater_than = ~(equal | less_than)
        exp[gene] = greater_than.astype(int)

    return exp


def get_cell_data(marker_path, tsne_path, cluster_path):
    """Parses cell data into a DataFrame, raising exceptions as necessary.
    Combines data at given paths into a single DataFrame. Assumes standard csv
    format, with each row corresponding to a single cell, and each column
    either to a single gene, a tSNE score, or a cluster identifier.
    Args:
        marker_path: Path to file containing gene expression data.
        tsne_path: Path to file containing tSNE score data.
        cluster_path: Path to file containing cluster data.
    Returns:
        A single pandas DataFrame incorporating all cell data. Row values are
        cell identifiers. The first column is cluster number. The 2nd and 3rd
        columns are tSNE_1 and tSNE_2 scores. All following columns are gene
        names, in the order given by the file at tsne_path.
    Raises:
        OSError: An error occurred accessing the files at marker_path,
            tsne_path, and cluster_path.
        ValueError: The files at marker_path, tsne_path, and cluster_path have
            inappropriate formatting.
        ValueError: The data contained in the files at marker_path, etc. is
            invalid.
    """

    cell_data = pd.read_csv(marker_path, index_col=0)
    cluster_data = pd.read_csv(cluster_path, header=None, index_col=0)
    # TODO: are these assignments necessary?
    cluster_data = cluster_data.rename(index=str, columns={1: "cluster"})
    tsne_data = pd.read_csv(tsne_path, header=None, index_col=0)
    tsne_data = tsne_data.rename(index=str, columns={1: "tSNE_1", 2: "tSNE_2"})
    cell_data = pd.concat(
        [cluster_data, tsne_data, cell_data], axis=1, sort=False
    ).rename_axis(None)
    # To get the complements of each gene, simply negate them. This makes
    # sorting and our algorithms work in the "opposite direction."
    for gene in cell_data.columns[3:]:
        cell_data[gene + "_c"] = -cell_data[gene]

    return cell_data


def singleton_test(cells, cluster, start=3, X=None, L=None):
    """Tests and ranks genes and complements using the XL-mHG test and t-test.
    Applies the XL-mHG test and t-test to each gene (and its complement) found
    in the columns of cells using the specified cluster of interest, then
    ranks genes by both mHG p-value and by t-test p-value. Specifically, the
    XL-mHG test calculates the expression cutoff value that best differentiates
    the cluster of interest from the cell population, then calculates the
    cluster's significance value using the hypergeometric and t-test. The final
    rank is the average of these two ranks.  Returns a pandas DataFrame with
    test and rank data by gene.
    Args:
        cells: A DataFrame with format matching those returned by
            get_cell_data. Row values are cell identifiers, columns are first
            cluster identifier, then tSNE_1 and tSNE_2, then gene names.
        cluster: The cluster of interest. Must be in cells['cluster'].values.
        X: A parameter of the XL-mHG test. An integer specifying the minimum
            number of cells in-cluster that should express the gene. Defaults
            to 15% of the cell count.
        L: A parameter of the XL-mHG test. An integer specifying the maximum
            number of cells that should express the gene. Defaults to 85% of
            the cell count.
    Returns:
        A single pandas DataFrame incorporating test and rank data, sorted
        by combined mHG and t rank. Row values are gene and complement names;
        complements are named by appending '_c' to gene names. Column values
        are:
            'HG_stat': The HG test statistic, measuring significance of the
                cluster at our given expression cutoff.
            'mHG_pval': The significance of our chosen cutoff.
            'mHG_cutoff_index': The expression cutoff index, for a list of all
                cells sorted by expression (descending for normal genes,
                ascending for their complements. Cells at the cutoff index or
                after do not express, while cells before the cutoff index
                express. The index starts at 0.
            'mHG_cutoff_value': Similar to 'mHG_cutoff_index'. The expression
                cutoff value. Cells above the cutoff express, while cells at or
                below the cutoff do not express.
            't_stat': The t-test statistic, measuring significance of the
                cluster at our given expression cutoff.
            't_pval': The p-value corresponding to the t-test statistic.
            'rank': The average of the gene's 'mHG_pval' rank and 't_pval'
                rank. Lower p-value is higher rank. A rank of 1 ("first") is
                higher than a rank of 2 ("second").
            'sequential_rank': Like rank, but goes from 1 to number of genes
                without repeats.
    Raises:
        ValueError: cells is in an incorrect format, cluster is not in
        cells['cluster'].values, X is greater than the cluster size or less
        than 0, or L is less than X or greater than the population size.
    """

    output = pd.DataFrame(columns=[
        'gene', 'HG_stat', 'mHG_pval', 'mHG_cutoff_index', 'mHG_cutoff_value',
        't_stat', 't_pval','w_stat', 'w_pval','ks_stat','ks_pval','enrichment_above_cutoff'
    ])
    for index, gene in enumerate(cells.columns[start:]):
        # Do XL-mHG and t-test for selected gene.
        # XL-mHG requires two list inputs, exp and v:
        # exp is a DataFrame with column values: cluster, gene expression.
        # Row values are cells.
        # v is boolean list of cluster membership. Uses 1s and 0s rather than
        # True and False.
        # Both exp and v are sorted by expression (descending).
        # print(
        #    "Testing " + str(gene) + "... [" + str(index + 1) + "/"
        #    + str(len(cells.columns[3:])) + "]"
        #)
        exp = cells[['cluster', gene]]
        exp = exp.sort_values(by=gene, ascending=False)
        v = np.array((exp['cluster'] == cluster).tolist()).astype(int)
        if X is None:
            X = 1
        if L is None:
            L = cells.shape[0]
        HG_stat, mHG_cutoff_index, mHG_pval = hg.xlmhg_test(v, X=X, L=L)
        mHG_cutoff_value = exp.iloc[mHG_cutoff_index][gene]
        # "Slide up" the cutoff index to where the gene expression actually
        # changes. I.e. for array [5.3 1.2 0 0 0 0], if index == 4, the cutoff
        # value is 0 and thus the index should actually slide to 2.
        # numpy.searchsorted works with ascending, not descending lists.
        # Therefore, negate to make it work.
        mHG_cutoff_index = np.searchsorted(
            -exp[gene].values, -mHG_cutoff_value, side='left'
        )
        # If index is 0 (all cells selected), slide to next value)
        if np.isclose(mHG_cutoff_index, 0):
            mHG_cutoff_index = np.searchsorted(
                -exp[gene].values, -mHG_cutoff_value, side='right'
            )
            if mHG_cutoff_index == exp.shape[0]:
                print(
                    "WARNING: "
                    + gene
                    + " has uniform expression across the population!"
                )
            else:
                mHG_cutoff_value = exp.iloc[mHG_cutoff_index][gene]
        # TODO: reassigning sample and population every time is inefficient
        sample = exp[exp['cluster'] == cluster]
        population = exp[exp['cluster'] != cluster]
        t_stat, t_pval = ss.ttest_ind(sample[gene], population[gene],equal_var=False)
        w_stat, w_pval = ss.ranksums(sample[gene], population[gene])
        ks_stat, ks_pval = ss.ks_2samp(sample[gene], population[gene])
        enrichment = np.sum(v[0:mHG_cutoff_index])/np.sum(v) # percentage of 1's above cutoff
        gene_data = pd.DataFrame(
            {
                'gene': gene, 'HG_stat': HG_stat, 'mHG_pval': mHG_pval,
                'mHG_cutoff_index': mHG_cutoff_index,
                'mHG_cutoff_value': mHG_cutoff_value,
                't_stat': t_stat, 't_pval': t_pval,
                'w_stat': w_stat, 'w_pval': w_pval,
                'ks_stat': ks_stat, 'ks_pval': ks_pval,
                'enrichment_above_cutoff': enrichment
            },
            index=[0]
        )
        output = output.append(gene_data, sort=False, ignore_index=True)

    # Pandas doesn't play nice with floating point fuzziness. Use fuzzy_rank to
    # deal with this.
    HG_rank = fuzzy_rank(output['HG_stat'])
    t_rank = fuzzy_rank(output['t_pval'])
    w_rank = fuzzy_rank(output['w_pval'])
    ks_rank = fuzzy_rank(output['ks_pval'])
    combined_rank = pd.DataFrame({
        'rank': HG_rank + t_rank + w_rank,
        'HG_rank': HG_rank,
        't_rank': t_rank,
        'w_rank': w_rank,
        'ks_rank': ks_rank
    })
    output = output.join(combined_rank)
    # Genes with the same rank will be sorted by HG statistic.
    # Mergesort is stable.
    output = output.sort_values(by='HG_rank', ascending=True)
    output = output.sort_values(by='rank', ascending=True, kind='mergesort')
    output['sequential_rank'] = output.reset_index().index + 1
    return output


def pair_test(cells, singleton, cluster, L=None, min_exp_ratio=0.4):
    """Tests and ranks pairs of genes using the hypergeometric test.
    Uses output of singleton_test to apply the hypergeometric test to pairs of
    genes, by finding the significance of the cluster of interest in the set of
    cells expressing both genes. Then, ranks pairs by this significance.
    Designed to use output of singleton_test. Output includes singleton data.
    Args:
        cells: A DataFrame with format matching those returned by
            get_cell_data. Row values are cell identifiers, columns are first
            cluster identifier, then tSNE_1 and tSNE_2, then gene names.
        singleton: A DataFrame with format matching those returned by
            singleton_test.
        cluster: The cluster of interest. Must be in cells['cluster'].values.
        min_exp_ratio: The minimum expression ratio for consideration. Genes
            which are expressed by a fraction of the cluster of interest less
            than this value are ignored. Usually these genes are unhelpful,
            and ignoring them helps with speed.
    Returns:
        A single pandas DataFrame incorporating test data, sorted by HG test
        statistic. Row values are unimportant. Column values are:
            'gene': The first gene in the pair.
            'gene_B': The second gene in the pair.
            'HG_stat': The HG test statistic, measuring significance of the
                cluster in the subset of cells which express both genes.
    Raises:
        ValueError: cells or singleton is in an incorrect format, cluster is
            not in cells['cluster'].values, min_exp_ratio is not in the range
            [0.0,1.0].
    """

    # create matrices with specifications:
    # cluster_product: each cell has row gene and column gene; value is number
    # of cells in cluster expressing both
    # not_cluster_product: same except counting number of cells not in cluster
    # expressing both
    if L is None:
        L = cells.shape[0]
    cluster_list = cells.iloc[:, 0]
    exp = get_discrete_exp(cells, singleton)
    cluster_exp = exp[cluster_list == cluster]
    not_cluster_exp = exp[cluster_list != cluster]
    total_num = exp.shape[0]
    cluster_size = cluster_exp.shape[0]
    # DROP COLUMNS WITH cluster expression below threshold FROM ONLY MULTI DFs
    # ALSO DROP COLUMNS with index>L
    dropped_genes = list([])
    for gene in cluster_exp.columns:
        if cluster_exp[gene].sum() / float(cluster_size) < min_exp_ratio:
            dropped_genes.append(gene)
        elif (
            singleton[singleton['gene'] == gene]['mHG_cutoff_index'].iloc[0]
            > L
        ):
            dropped_genes.append(gene)

    gene_list = np.setdiff1d(cluster_exp.columns.values, dropped_genes)
    print(
        "Dropped " + str(len(dropped_genes))
        + " genes from multiple gene testing. Threshold is "
        + str(min_exp_ratio)
    )
    cluster_product = cluster_exp.T.dot(cluster_exp)
    not_cluster_product = not_cluster_exp.T.dot(not_cluster_exp)

    output = pd.DataFrame(columns=['HG_stat'])
    gene_count = len(gene_list)
    for i in range(0, gene_count):
        gene_A = gene_list[i]
        print(
            "[" + str(i+1) + "/" + str(gene_count) + "] Pairing "
            + gene_A + "..."
        )
        # start gene_B iterating from index i+1 to prevent duplicate pairs
        for gene_B in gene_list[i+1:]:
            # For middle expression: pair gene with complement
            num_exp_in_cluster = cluster_product.loc[gene_A, gene_B]
            num_expressed = (
                num_exp_in_cluster +
                not_cluster_product.loc[gene_A, gene_B]
            )
            # skip if no intersection exists
            if num_expressed == 0:
                continue

            HG_stat = ss.hypergeom.sf(
                num_exp_in_cluster,
                total_num,
                cluster_size,
                num_expressed,
                loc=1
            )
            gene_df = pd.DataFrame(
                {'HG_stat': HG_stat, 'gene': gene_A, 'gene_B': gene_B},
                index=[0]
            )
            output = output.append(gene_df, sort=False, ignore_index=True)

    output = singleton.append(output, sort=False, ignore_index=True)
    # rerank by HG statistic and sort by HG only
    # fuzzy_rank to avoid pandas issues with fuzziness
    HG_rank = fuzzy_rank(output['HG_stat'])
    output['HG_rank'] = HG_rank
    output = output.sort_values(by='HG_rank', ascending=True)
    # put gene_B in the 2nd place
    output = output[['gene'] + ['gene_B'] + output.columns.tolist()[1:-1]]
    output['sequential_rank'] = output.reset_index().index + 1
    return output


def find_TP_TN_singleton(cells, singleton, cluster,start=3):
    """Finds true positive/true negative rates, appends them to our DataFrames.
    Uses output of singleton_test to find true positive and true
    negative rates for gene expression relative to the cluster of interest.
    Appends rates to these output DataFrames.
    Args:
        cells: A DataFrame with format matching those returned by
            get_cell_data. Row values are cell identifiers, columns are first
            cluster identifier, then tSNE_1 and tSNE_2, then gene names.
        singleton: A DataFrame with format matching those returned by
            singleton_test.
        cluster: The cluster of interest. Must be in cells['cluster'].values.
    Returns:
        singleton with columns 'true_positive' and 'true_negative'
        added.
    Raises:
        ValueError: cells or singleton is in an incorrect format, or
            cluster is not in cells['cluster'].values.
    """

    # TODO: column operations for efficiency
    # TODO: stop code reuse
    exp = get_discrete_exp(cells, singleton,start)
    cluster_list = cells['cluster']
    cluster_size = cluster_list[cluster_list == cluster].size
    cluster_exp = exp[cluster_list == cluster]
    not_cluster_not_exp = 1 - exp[cluster_list != cluster]
    TP_TN_singleton = pd.DataFrame()
    i = 0
    for index, row in singleton.iterrows():
        i += 1
        if i % 500 == 0:
            print(str(i) + " of " + str(singleton.shape[0]))
        gene = row['gene']
        true_positives = cluster_exp[gene].sum()
        positives = cluster_size
        true_negatives = not_cluster_not_exp[gene].sum()
        negatives = cells.shape[0] - positives
        TP_rate = true_positives / float(positives)
        TN_rate = true_negatives / float(negatives)

        row_df = pd.DataFrame(
            {
                'gene': gene, 'true_positive': TP_rate,
                'true_negative': TN_rate
            }, index=[0]
        )
        TP_TN_singleton = TP_TN_singleton.append(row_df, sort=False, ignore_index=True)
        singleton = singleton.merge(TP_TN_singleton, how='left')


    return TP_TN_singleton

def find_TP_TN(cells, singleton, pair, cluster):
    """Finds true positive/true negative rates, appends them to our DataFrames.
    Uses output of singleton_test and pair_test to find true positive and true
    negative rates for gene expression relative to the cluster of interest.
    Appends rates to these output DataFrames.
    Args:
        cells: A DataFrame with format matching those returned by
            get_cell_data. Row values are cell identifiers, columns are first
            cluster identifier, then tSNE_1 and tSNE_2, then gene names.
        singleton: A DataFrame with format matching those returned by
            singleton_test.
        pair: A DataFrame with format matching those returned by pair_test.
        cluster: The cluster of interest. Must be in cells['cluster'].values.
    Returns:
        singleton and pair with columns 'true_positive' and 'true_negative'
        added.
    Raises:
        ValueError: cells, singleton, or pair is in an incorrect format, or
            cluster is not in cells['cluster'].values.
    """

    # TODO: column operations for efficiency
    # TODO: stop code reuse
    exp = get_discrete_exp(cells, singleton)
    cluster_list = cells['cluster']
    cluster_size = cluster_list[cluster_list == cluster].size
    cluster_exp = exp[cluster_list == cluster]
    not_cluster_not_exp = 1 - exp[cluster_list != cluster]
    TP_TN = pd.DataFrame()
    TP_TN_singleton = pd.DataFrame()
    i = 0
    for index, row in pair.iterrows():
        i += 1
        if i % 500 == 0:
            print(str(i) + " of " + str(pair.shape[0]))
        gene = row['gene']
        gene_B = row['gene_B']
        if pd.isnull(gene_B):
            true_positives = cluster_exp[gene].sum()
            positives = cluster_size
            true_negatives = not_cluster_not_exp[gene].sum()
            negatives = cells.shape[0] - positives
        elif pd.notnull(gene_B):
            true_positives = cluster_exp[
                cluster_exp[gene_B] == 1
            ][gene].sum()
            positives = cluster_size
            true_negatives = not_cluster_not_exp[
                not_cluster_not_exp[gene] + not_cluster_not_exp[gene_B] > 0
            ].shape[0]
            negatives = cells.shape[0] - positives
        TP_rate = true_positives / float(positives)
        TN_rate = true_negatives / float(negatives)
        if pd.isnull(gene_B):
            row_df_singleton = pd.DataFrame(
                {
                    'gene': gene, 'true_positive': TP_rate,
                    'true_negative': TN_rate
                }, index=[0]
            )
            TP_TN_singleton = TP_TN_singleton.append(
                row_df_singleton, sort=False, ignore_index=True
            )

        row_df = pd.DataFrame(
            {
                'gene': gene, 'gene_B': gene_B, 'true_positive': TP_rate,
                'true_negative': TN_rate
            }, index=[0]
        )
        TP_TN = TP_TN.append(row_df, sort=False, ignore_index=True)

    pair = pair.merge(TP_TN, how='left')
    # index is pair index; fix this for singleton
    singleton = singleton.merge(TP_TN_singleton, how='left')
    return singleton, pair

def find_weighted_TP_TN_singleton(cells, singleton, cluster):
    """Finds TP/TN rates weighted by gene quality.
    Uses output of singleton_test to find weighted TP/TN rates
    for gene expression weighted by gene quality. Gene quality is defined as
    the number of cells which have nonzero expression. Two methods are
    used. First, the gene list is split into 3 buckets by quality, and their
    TP/TN are weighted by fixed coefficients (i.e. 1, 2, 3). Second, the gene
    list is split into 6 buckets and their TP/TN are weighted by median quality
    of the bucket.
    Args:
        cells: A DataFrame with format matching those returned by
            get_cell_data. Row values are cell identifiers, columns are first
            cluster identifier, then tSNE_1 and tSNE_2, then gene names, then
            true positive and true negative, unweighted.
        singleton: A DataFrame with format matching those returned by
            singleton_test.
        cluster: The cluster of interest. Must be in cells['cluster'].values.
    Returns:
        singleton with columns 'weighted_TP_1', 'weighted_TN_1',
        'weight_1', 'weighted_TP_2', 'weighted_TN_2', 'weight_2', 'quality'.
    Raises:
        ValueError: cells or singleton is in an incorrect format, or
            cluster is not in cells['cluster'].values.
    """

    QUANTILES_METHOD_1 = 3
    QUANTILES_METHOD_2 = 6
    quality = (cells.astype(bool).sum(axis=0) /
               cells.shape[0]).rename('quality')
    singleton = singleton.join(quality, on='gene', sort=False)
    singleton['weight_1'] = (
        (
            pd.qcut(
                singleton['quality'], QUANTILES_METHOD_1,
                labels=False, duplicates='drop'
            ) + 1
        ) / QUANTILES_METHOD_1
    )
    singleton['quantile'] = (
        pd.qcut(
            singleton['quality'], QUANTILES_METHOD_2,
            labels=False, duplicates='drop'
        ) + 1
    )
    singleton['weight_2'] = (
        singleton[
            ['quality', 'quantile']
        ].groupby('quantile').transform('median')
    )
    singleton['weighted_TP_1'] = (
        singleton['true_positive'] * singleton['weight_1']
    )
    singleton['weighted_TN_1'] = (
        singleton['true_negative'] * singleton['weight_1']
    )
    singleton['weighted_TP_2'] = (
        singleton['true_positive'] * singleton['weight_2']
    )
    singleton['weighted_TN_2'] = (
        singleton['true_negative'] * singleton['weight_2']
    )

    return singleton

def find_weighted_TP_TN(cells, singleton, pair, cluster):
    """Finds TP/TN rates weighted by gene quality.
    Uses output of singleton_test and pair_test to find weighted TP/TN rates
    for gene expression weighted by gene quality. Gene quality is defined as
    the number of cells which have nonzero expression. Two methods are
    used. First, the gene list is split into 3 buckets by quality, and their
    TP/TN are weighted by fixed coefficients (i.e. 1, 2, 3). Second, the gene
    list is split into 6 buckets and their TP/TN are weighted by median quality
    of the bucket.
    Args:
        cells: A DataFrame with format matching those returned by
            get_cell_data. Row values are cell identifiers, columns are first
            cluster identifier, then tSNE_1 and tSNE_2, then gene names, then
            true positive and true negative, unweighted.
        singleton: A DataFrame with format matching those returned by
            singleton_test.
        pair: A DataFrame with format matching those returned by pair_test.
        cluster: The cluster of interest. Must be in cells['cluster'].values.
    Returns:
        singleton and pair with columns 'weighted_TP_1', 'weighted_TN_1',
        'weight_1', 'weighted_TP_2', 'weighted_TN_2', 'weight_2', 'quality'.
    Raises:
        ValueError: cells, singleton, or pair is in an incorrect format, or
            cluster is not in cells['cluster'].values.
    """

    QUANTILES_METHOD_1 = 3
    QUANTILES_METHOD_2 = 6
    quality = (cells.astype(bool).sum(axis=0) /
               cells.shape[0]).rename('quality')
    singleton = singleton.join(quality, on='gene', sort=False)
    singleton['weight_1'] = (
        (
            pd.qcut(
                singleton['quality'], QUANTILES_METHOD_1,
                labels=False, duplicates='drop'
            ) + 1
        ) / QUANTILES_METHOD_1
    )
    singleton['quantile'] = (
        pd.qcut(
            singleton['quality'], QUANTILES_METHOD_2,
            labels=False, duplicates='drop'
        ) + 1
    )
    singleton['weight_2'] = (
        singleton[
            ['quality', 'quantile']
        ].groupby('quantile').transform('median')
    )
    singleton['weighted_TP_1'] = (
        singleton['true_positive'] * singleton['weight_1']
    )
    singleton['weighted_TN_1'] = (
        singleton['true_negative'] * singleton['weight_1']
    )
    singleton['weighted_TP_2'] = (
        singleton['true_positive'] * singleton['weight_2']
    )
    singleton['weighted_TN_2'] = (
        singleton['true_negative'] * singleton['weight_2']
    )

    # get pair weights by averaging weights of individuals
    pair_only = pair[pair['gene_B'].notnull()]
    pair_weight = pair_only[['gene', 'gene_B']]
    pair_weight = pair_weight.merge(
        singleton[['gene', 'weight_1', 'weight_2']],
        left_on='gene',
        right_on='gene',
        how='left'
    )
    pair_weight = pair_weight.rename(
        index=str,
        columns={
            'weight_1': 'gene_weight_1',
            'weight_2': 'gene_weight_2'
        }
    )
    pair_weight = pair_weight.merge(
        singleton[['gene', 'weight_1', 'weight_2']],
        left_on='gene_B',
        right_on='gene',
        how='left'
    )
    pair_weight = pair_weight.rename(
        index=str,
        columns={
            'gene_x': 'gene',
            'weight_1': 'gene_B_weight_1',
            'weight_2': 'gene_B_weight_2'
        }
    ).drop(columns='gene_y')
    pair_weight['weight_1'] = (
        (pair_weight['gene_weight_1'] + pair_weight['gene_B_weight_1'])
        / 2
    )
    pair_weight['weight_2'] = (
        (pair_weight['gene_weight_2'] + pair_weight['gene_B_weight_2'])
        / 2
    )
    pair_weight = pair_weight.append(
        singleton[['gene', 'weight_1', 'weight_2']], ignore_index=True,
        sort=False
    )
    pair = pair.merge(
        pair_weight[['gene', 'gene_B', 'weight_1', 'weight_2']],
        on=['gene', 'gene_B'],
        how='left',
        sort=False
    )
    pair['weighted_TP_1'] = (
        pair['true_positive'] * pair['weight_1']
    )
    pair['weighted_TN_1'] = (
        pair['true_negative'] * pair['weight_1']
    )
    pair['weighted_TP_2'] = (
        pair['true_positive'] * pair['weight_2']
    )
    pair['weighted_TN_2'] = (
        pair['true_negative'] * pair['weight_2']
    )
    return singleton, pair



#####################################################
########### HIGH RES PLOT CONSTRUCTION ###############
#####################################################


def make_discrete_plots(cells, singleton, pair, plot_pages, path):
    """Creates plots of discrete expression and saves them to pdf.
    Uses output of singleton_test and pair_test to create plots of discrete
    gene expression. For gene pairs, plot each individual gene as well,
    comparing their plots to the pair plot. Plots most significant genes/pairs
    first.
    Args:
        cells: A DataFrame with format matching those returned by
            get_cell_data. Row values are cell identifiers, columns are first
            cluster identifier, then tSNE_1 and tSNE_2, then gene names.
        singleton: A DataFrame with format matching those returned by
            singleton_test.
        pair: A DataFrame with format matching those returned by pair_test.
        plot_pages: The maximum number of pages to plot. Each gene or gene pair
            corresponds to one page.
        path: Save the plot pdf here.
    Returns:
        Nothing.
    Raises:
        ValueError: cells, singleton, or pair is in an incorrect format,
            plot_pages is less than 1.
    """

    exp = get_discrete_exp(cells, singleton)
    with PdfPages(path) as pdf:
        for i in range(0, plot_pages):
            print("Plotting discrete plot "+str(i+1)+" of "+str(plot_pages))
            gene_A = pair['gene'].iloc[i]
            gene_B = pair['gene_B'].iloc[i]

            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))

            if pd.isnull(gene_B):
                c = (exp[gene_A] == 1)
                ax1.set_title("rank " + str(i+1) + ": " + gene_A)
            else:
                c = (exp[gene_A] == 1) & (exp[gene_B] == 1)
                ax1.set_title(
                    "rank " + str(i+1) + ": " + gene_A + "+" + gene_B
                )

            sc1 = ax1.scatter(
                x=cells['tSNE_1'],
                y=cells['tSNE_2'],
                s=2,
                c=c,
                cmap=cm.get_cmap('bwr')
            )
            ax1.set_xlabel("tSNE_1")
            ax1.set_ylabel("tSNE_2")
            # plt.colorbar(sc1, ax=ax1)

            if not pd.isnull(gene_B):
                sc2 = ax2.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=exp[gene_A],
                    cmap=cm.get_cmap('bwr')
                )
                ax2.set_xlabel("tSNE_1")
                ax2.set_ylabel("tSNE_2")
                ax2.set_title(
                    gene_A + " %.3f" %
                    np.absolute(pair[
                        (pair['gene'] == gene_A) & (pair['gene_B'].isnull())
                    ]['mHG_cutoff_value'].iloc[0])
                )
                # plt.colorbar(sc2, ax=ax2)

                sc3 = ax3.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=exp[gene_B],
                    cmap=cm.get_cmap('bwr')
                )
                ax3.set_xlabel("tSNE_1")
                ax3.set_ylabel("tSNE_2")
                ax3.set_title(
                    gene_B + " %.3f" %
                    np.absolute(pair[
                        (pair['gene'] == gene_B) & (pair['gene_B'].isnull())
                    ]['mHG_cutoff_value'].iloc[0])
                )
                # plt.colorbar(sc3, ax=ax3)

            pdf.savefig(fig)
            plt.close(fig)


def make_combined_plots(
    cells, singleton, pair, plot_pages, pair_path, singleton_path
):
    """Creates a plot of discrete and continuous expression and saves to pdf.
    Uses output of singleton_test and pair_test to create plots of discrete and
    continuous expression. For gene pairs, make these two plots for each gene
    in the pair. Plots most significant genes/pairs first. Also makes a similar
    plot using only singletons.
    Args:
        cells: A DataFrame with format matching those returned by
            get_cell_data. Row values are cell identifiers, columns are first
            cluster identifier, then tSNE_1 and tSNE_2, then gene names.
        singleton: A DataFrame with format matching those returned by
            singleton_test.
        pair: A DataFrame with format matching those returned by pair_test.
        plot_pages: The maximum number of pages to plot. Each gene or gene pair
            corresponds to one page.
        pair_path: Save the full plot pdf here.
        singleton_path: Save the singleton-only plot pdf here.
    Returns:
        Nothing.
    Raises:
        ValueError: cells, singleton, or pair is in an incorrect format,
            plot_pages is less than 1.
    """

    CMAP_CONTINUOUS = cm.get_cmap('nipy_spectral')
    CMAP_DISCRETE = cm.get_cmap('bwr')
    exp = get_discrete_exp(cells, singleton)
    with PdfPages(pair_path) as pdf:
        for i in range(0, plot_pages):
            print("Plotting combination plot "+str(i+1)+" of "+str(plot_pages))
            # plot the cutoff
            gene_A = pair['gene'].iloc[i]
            gene_B = pair['gene_B'].iloc[i]

            # Plot regular gene instead of complement.
            # This is unnecessary and counterproductive.
            """
            pattern = re.compile(r"^(.*)_c+$")
            search_A = re.search(pattern, gene_A)
            if search_A:
                gene_A = search_A.group(1)
                if pd.notnull(gene_B):
                    search_B = re.search(pattern, gene_B)
                    if search_B:
                        gene_B = search_B.group(1)
            """

            # Graphs twogene and singlegene differently.
            if not pd.isnull(gene_B):
                fig, ((ax1a, ax1b), (ax2a, ax2b)) = plt.subplots(
                    nrows=2, ncols=2, figsize=(10, 10)
                )

                sc1a = ax1a.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=exp[gene_A],
                    cmap=CMAP_DISCRETE
                )
                ax1a.set_xlabel("tSNE_1")
                ax1a.set_ylabel("tSNE_2")
                ax1a.set_title(
                    gene_A + " %.3f" %
                    np.absolute(pair[
                        (pair['gene'] == gene_A) & (pair['gene_B'].isnull())
                    ]['mHG_cutoff_value'].iloc[0])
                )
                # plt.colorbar(sc1a, ax=ax1a)

                sc1b = ax1b.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=np.absolute(cells[gene_A]),
                    cmap=CMAP_CONTINUOUS
                )
                ax1b.set_xlabel("tSNE_1")
                ax1b.set_ylabel("tSNE_2")
                ax1b.set_title(gene_A)
                plt.colorbar(sc1b, ax=ax1b)

                sc2a = ax2a.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=exp[gene_B],
                    cmap=CMAP_DISCRETE
                )
                ax2a.set_xlabel("tSNE_1")
                ax2a.set_ylabel("tSNE_2")
                ax2a.set_title(
                    gene_B + " %.3f" %
                    np.absolute(pair[
                        (pair['gene'] == gene_B) & (pair['gene_B'].isnull())
                    ]['mHG_cutoff_value'].iloc[0])
                )
                # plt.colorbar(sc2a, ax=ax2a)

                sc2b = ax2b.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=np.absolute(cells[gene_B]),
                    cmap=CMAP_CONTINUOUS
                )
                ax2b.set_xlabel("tSNE_1")
                ax2b.set_ylabel("tSNE_2")
                ax2b.set_title(gene_B)
                plt.colorbar(sc2b, ax=ax2b)
            else:
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

                sc1 = ax1.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=exp[gene_A],
                    cmap=CMAP_DISCRETE
                )
                ax1.set_xlabel("tSNE_1")
                ax1.set_ylabel("tSNE_2")
                ax1.set_title(
                    gene_A + " %.3f" %
                    np.absolute(pair[
                        (pair['gene'] == gene_A) & (pair['gene_B'].isnull())
                    ]['mHG_cutoff_value'].iloc[0])
                )
                # plt.colorbar(sc1, ax=ax1)

                sc2 = ax2.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=np.absolute(cells[gene_A]),
                    cmap=CMAP_CONTINUOUS
                )
                ax2.set_xlabel("tSNE_1")
                ax2.set_ylabel("tSNE_2")
                ax2.set_title(gene_A)
                plt.colorbar(sc2, ax=ax2)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(singleton_path) as pdf:
        for i in range(0, plot_pages):
            print(
                "Plotting single combination plot " + str(i+1)
                + " of " + str(plot_pages)
            )

            # plot the cutoff
            gene = singleton['gene'].iloc[i]

            # Plot regular gene instead of complement.
            # This is unnecessary and counterproductive.
            """
            pattern = re.compile(r"^(.*)_c+$")
            search = re.search(pattern, gene)
            if search:
                gene = search.group(1)
            """

            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

            sc1 = ax1.scatter(
                x=cells['tSNE_1'],
                y=cells['tSNE_2'],
                s=3,
                c=exp[gene],
                cmap=CMAP_DISCRETE
            )
            ax1.set_xlabel("tSNE_1")
            ax1.set_ylabel("tSNE_2")
            ax1.set_title(
                gene + " %.3f" %
                np.absolute(singleton[
                    singleton['gene'] == gene
                ]['mHG_cutoff_value'].iloc[0])
            )
            # plt.colorbar(sc1, ax=ax1)

            sc2 = ax2.scatter(
                x=cells['tSNE_1'],
                y=cells['tSNE_2'],
                s=3,
                c=np.absolute(cells[gene]),
                cmap=CMAP_CONTINUOUS
            )
            ax2.set_xlabel("tSNE_1")
            ax2.set_ylabel("tSNE_2")
            ax2.set_title(gene)
            plt.colorbar(sc2, ax=ax2)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def make_TP_TN_plots(
    cells, singleton, pair, plot_genes, pair_path, singleton_path
):
    """Creates plots of true positive/true negative rates and saves to pdf.
    Uses output of find_TP_TN to create plots of true positive and true
    negative rate by gene or gene pair. Plots most significant genes/pairs
    first. Also makes a similar plot using only singletons.
    Args:
        cells: A DataFrame with format matching those returned by
        get_cell_data. Row values are cell identifiers, columns are first
        cluster identifier, then tSNE_1 and tSNE_2, then gene names.
        singleton: A DataFrame with format matching those returned by
        singleton_test.
        pair: A DataFrame with format matching those returned by pair_test.
        plot_genes: Number of genes/pairs to plot. More is messier!
        pair_path: Save the full plot pdf here.
        singleton_path: Save the singleton-only plot pdf here.
    Returns:
        Nothing.
    Raises:
        ValueError: cells, singleton, or pair is in an incorrect format,
        plot_pages is less than 1.
    """
    PADDING = 0.002

    fig = plt.figure(figsize=[15, 15])
    plt.xlabel("True positive")
    plt.ylabel("True negative")
    plt.title("True positive/negative")
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.scatter(pair.iloc[:20]['true_positive'],
                pair.iloc[:20]['true_negative'],
                s=3)

    for i in range(0, 20):
        row = pair.iloc[i]
        if pd.isnull(row['gene_B']):
            plt.annotate(
                row['gene'], (row['true_positive'] + PADDING,
                              row['true_negative'] + PADDING),
            )
        else:
            plt.annotate(
                row['gene'] + "+" + row['gene_B'],
                (row['true_positive'] + PADDING,
                 row['true_negative'] + PADDING)
            )

    fig.savefig(pair_path)
    plt.close(fig)

    fig = plt.figure(figsize=[15, 15])
    plt.xlabel("True positive")
    plt.ylabel("True negative")
    plt.title("True positive/negative")
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.scatter(singleton.iloc[:20]['true_positive'],
                singleton.iloc[:20]['true_negative'],
                s=3)

    for i in range(0, 20):
        row = singleton.iloc[i]
        plt.annotate(row['gene'], (row['true_positive'] +
                                   PADDING, row['true_negative'] + PADDING))

    fig.savefig(singleton_path)
    plt.close(fig)



#####################################################
########### LOW RES PLOT CONSTRUCTION ###############
#####################################################


def make_discrete_plots_low(cells, singleton, pair, plot_pages, path):
    """Creates plots of discrete expression and saves them to pdf.
    Uses output of singleton_test and pair_test to create plots of discrete
    gene expression. For gene pairs, plot each individual gene as well,
    comparing their plots to the pair plot. Plots most significant genes/pairs
    first.
    Args:
        cells: A DataFrame with format matching those returned by
            get_cell_data. Row values are cell identifiers, columns are first
            cluster identifier, then tSNE_1 and tSNE_2, then gene names.
        singleton: A DataFrame with format matching those returned by
            singleton_test.
        pair: A DataFrame with format matching those returned by pair_test.
        plot_pages: The maximum number of pages to plot. Each gene or gene pair
            corresponds to one page.
        path: Save the plot pdf here.
    Returns:
        Nothing.
    Raises:
        ValueError: cells, singleton, or pair is in an incorrect format,
            plot_pages is less than 1.
    """

    exp = get_discrete_exp(cells, singleton)
    with PdfPages(path) as pdf:
        for i in range(0, plot_pages):
            print("Plotting discrete plot "+str(i+1)+" of "+str(plot_pages))
            gene_A = pair['gene'].iloc[i]
            gene_B = pair['gene_B'].iloc[i]

            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))

            if pd.isnull(gene_B):
                c = (exp[gene_A] == 1)
                ax1.set_title("rank " + str(i+1) + ": " + gene_A)
            else:
                c = (exp[gene_A] == 1) & (exp[gene_B] == 1)
                ax1.set_title(
                    "rank " + str(i+1) + ": " + gene_A + "+" + gene_B
                )

            sc1 = ax1.scatter(
                x=cells['tSNE_1'],
                y=cells['tSNE_2'],
                s=2,
                c=c,
                cmap=cm.get_cmap('bwr')
            )
            ax1.set_xlabel("tSNE_1")
            ax1.set_ylabel("tSNE_2")
            # plt.colorbar(sc1, ax=ax1)

            if not pd.isnull(gene_B):
                sc2 = ax2.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=exp[gene_A],
                    cmap=cm.get_cmap('bwr')
                )
                ax2.set_xlabel("tSNE_1")
                ax2.set_ylabel("tSNE_2")
                ax2.set_title(
                    gene_A + " %.3f" %
                    np.absolute(pair[
                        (pair['gene'] == gene_A) & (pair['gene_B'].isnull())
                    ]['mHG_cutoff_value'].iloc[0])
                )
                # plt.colorbar(sc2, ax=ax2)

                sc3 = ax3.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=exp[gene_B],
                    cmap=cm.get_cmap('bwr')
                )
                ax3.set_xlabel("tSNE_1")
                ax3.set_ylabel("tSNE_2")
                ax3.set_title(
                    gene_B + " %.3f" %
                    np.absolute(pair[
                        (pair['gene'] == gene_B) & (pair['gene_B'].isnull())
                    ]['mHG_cutoff_value'].iloc[0])
                )
                # plt.colorbar(sc3, ax=ax3)

            pdf.savefig(fig)
            plt.close(fig)


def make_combined_plots_low(
    cells, singleton, pair, plot_pages, pair_path, singleton_path
):
    """Creates a plot of discrete and continuous expression and saves to pdf.
    Uses output of singleton_test and pair_test to create plots of discrete and
    continuous expression. For gene pairs, make these two plots for each gene
    in the pair. Plots most significant genes/pairs first. Also makes a similar
    plot using only singletons.
    Args:
        cells: A DataFrame with format matching those returned by
            get_cell_data. Row values are cell identifiers, columns are first
            cluster identifier, then tSNE_1 and tSNE_2, then gene names.
        singleton: A DataFrame with format matching those returned by
            singleton_test.
        pair: A DataFrame with format matching those returned by pair_test.
        plot_pages: The maximum number of pages to plot. Each gene or gene pair
            corresponds to one page.
        pair_path: Save the full plot pdf here.
        singleton_path: Save the singleton-only plot pdf here.
    Returns:
        Nothing.
    Raises:
        ValueError: cells, singleton, or pair is in an incorrect format,
            plot_pages is less than 1.
    """

    CMAP_CONTINUOUS = cm.get_cmap('nipy_spectral')
    CMAP_DISCRETE = cm.get_cmap('bwr')
    exp = get_discrete_exp(cells, singleton)
    with PdfPages(pair_path) as pdf:
        for i in range(0, plot_pages):
            print("Plotting combination plot "+str(i+1)+" of "+str(plot_pages))
            # plot the cutoff
            gene_A = pair['gene'].iloc[i]
            gene_B = pair['gene_B'].iloc[i]

            # Plot regular gene instead of complement.
            # This is unnecessary and counterproductive.
            """
            pattern = re.compile(r"^(.*)_c+$")
            search_A = re.search(pattern, gene_A)
            if search_A:
                gene_A = search_A.group(1)
                if pd.notnull(gene_B):
                    search_B = re.search(pattern, gene_B)
                    if search_B:
                        gene_B = search_B.group(1)
            """

            # Graphs twogene and singlegene differently.
            if not pd.isnull(gene_B):
                fig, ((ax1a, ax1b), (ax2a, ax2b)) = plt.subplots(
                    nrows=2, ncols=2, figsize=(10, 10)
                )

                sc1a = ax1a.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=exp[gene_A],
                    cmap=CMAP_DISCRETE
                )
                ax1a.set_xlabel("tSNE_1")
                ax1a.set_ylabel("tSNE_2")
                ax1a.set_title(
                    gene_A + " %.3f" %
                    np.absolute(pair[
                        (pair['gene'] == gene_A) & (pair['gene_B'].isnull())
                    ]['mHG_cutoff_value'].iloc[0])
                )
                # plt.colorbar(sc1a, ax=ax1a)

                sc1b = ax1b.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=np.absolute(cells[gene_A]),
                    cmap=CMAP_CONTINUOUS
                )
                ax1b.set_xlabel("tSNE_1")
                ax1b.set_ylabel("tSNE_2")
                ax1b.set_title(gene_A)
                plt.colorbar(sc1b, ax=ax1b)

                sc2a = ax2a.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=exp[gene_B],
                    cmap=CMAP_DISCRETE
                )
                ax2a.set_xlabel("tSNE_1")
                ax2a.set_ylabel("tSNE_2")
                ax2a.set_title(
                    gene_B + " %.3f" %
                    np.absolute(pair[
                        (pair['gene'] == gene_B) & (pair['gene_B'].isnull())
                    ]['mHG_cutoff_value'].iloc[0])
                )
                # plt.colorbar(sc2a, ax=ax2a)

                sc2b = ax2b.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=np.absolute(cells[gene_B]),
                    cmap=CMAP_CONTINUOUS
                )
                ax2b.set_xlabel("tSNE_1")
                ax2b.set_ylabel("tSNE_2")
                ax2b.set_title(gene_B)
                plt.colorbar(sc2b, ax=ax2b)
            else:
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

                sc1 = ax1.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=exp[gene_A],
                    cmap=CMAP_DISCRETE
                )
                ax1.set_xlabel("tSNE_1")
                ax1.set_ylabel("tSNE_2")
                ax1.set_title(
                    gene_A + " %.3f" %
                    np.absolute(pair[
                        (pair['gene'] == gene_A) & (pair['gene_B'].isnull())
                    ]['mHG_cutoff_value'].iloc[0])
                )
                # plt.colorbar(sc1, ax=ax1)

                sc2 = ax2.scatter(
                    x=cells['tSNE_1'],
                    y=cells['tSNE_2'],
                    s=3,
                    c=np.absolute(cells[gene_A]),
                    cmap=CMAP_CONTINUOUS
                )
                ax2.set_xlabel("tSNE_1")
                ax2.set_ylabel("tSNE_2")
                ax2.set_title(gene_A)
                plt.colorbar(sc2, ax=ax2)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(singleton_path) as pdf:
        for i in range(0, plot_pages):
            print(
                "Plotting single combination plot " + str(i+1)
                + " of " + str(plot_pages)
            )

            # plot the cutoff
            gene = singleton['gene'].iloc[i]

            # Plot regular gene instead of complement.
            # This is unnecessary and counterproductive.
            """
            pattern = re.compile(r"^(.*)_c+$")
            search = re.search(pattern, gene)
            if search:
                gene = search.group(1)
            """

            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

            sc1 = ax1.scatter(
                x=cells['tSNE_1'],
                y=cells['tSNE_2'],
                s=3,
                c=exp[gene],
                cmap=CMAP_DISCRETE
            )
            ax1.set_xlabel("tSNE_1")
            ax1.set_ylabel("tSNE_2")
            ax1.set_title(
                gene + " %.3f" %
                np.absolute(singleton[
                    singleton['gene'] == gene
                ]['mHG_cutoff_value'].iloc[0])
            )
            # plt.colorbar(sc1, ax=ax1)

            sc2 = ax2.scatter(
                x=cells['tSNE_1'],
                y=cells['tSNE_2'],
                s=3,
                c=np.absolute(cells[gene]),
                cmap=CMAP_CONTINUOUS
            )
            ax2.set_xlabel("tSNE_1")
            ax2.set_ylabel("tSNE_2")
            ax2.set_title(gene)
            plt.colorbar(sc2, ax=ax2)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def make_TP_TN_plots_low(
    cells, singleton, pair, plot_genes, pair_path, singleton_path
):
    """Creates plots of true positive/true negative rates and saves to pdf.
    Uses output of find_TP_TN to create plots of true positive and true
    negative rate by gene or gene pair. Plots most significant genes/pairs
    first. Also makes a similar plot using only singletons.
    Args:
        cells: A DataFrame with format matching those returned by
        get_cell_data. Row values are cell identifiers, columns are first
        cluster identifier, then tSNE_1 and tSNE_2, then gene names.
        singleton: A DataFrame with format matching those returned by
        singleton_test.
        pair: A DataFrame with format matching those returned by pair_test.
        plot_genes: Number of genes/pairs to plot. More is messier!
        pair_path: Save the full plot pdf here.
        singleton_path: Save the singleton-only plot pdf here.
    Returns:
        Nothing.
    Raises:
        ValueError: cells, singleton, or pair is in an incorrect format,
        plot_pages is less than 1.
    """
    PADDING = 0.002

    fig = plt.figure(figsize=[15, 15])
    plt.xlabel("True positive")
    plt.ylabel("True negative")
    plt.title("True positive/negative")
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.scatter(pair.iloc[:20]['true_positive'],
                pair.iloc[:20]['true_negative'],
                s=3)

    for i in range(0, 20):
        row = pair.iloc[i]
        if pd.isnull(row['gene_B']):
            plt.annotate(
                row['gene'], (row['true_positive'] + PADDING,
                              row['true_negative'] + PADDING),
            )
        else:
            plt.annotate(
                row['gene'] + "+" + row['gene_B'],
                (row['true_positive'] + PADDING,
                 row['true_negative'] + PADDING)
            )

    fig.savefig(pair_path)
    plt.close(fig)

    fig = plt.figure(figsize=[15, 15])
    plt.xlabel("True positive")
    plt.ylabel("True negative")
    plt.title("True positive/negative")
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.scatter(singleton.iloc[:20]['true_positive'],
                singleton.iloc[:20]['true_negative'],
                s=3)

    for i in range(0, 20):
        row = singleton.iloc[i]
        plt.annotate(row['gene'], (row['true_positive'] +
                                   PADDING, row['true_negative'] + PADDING))

    fig.savefig(singleton_path)
    plt.close(fig)

