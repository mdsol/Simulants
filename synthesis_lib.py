#!/usr/bin/env python
# coding: utf-8

# Author: Mandis Beigi
# Copyright (c) 2022 Medidata Solutions, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import numpy as np
import pandas as pd
import gower
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2_contingency
from scipy import stats
import logging
import math

import dimanalysis_lib
import preprocessor_lib
import utilities_lib


def is_in_group(col_name, col_groups):
    for col_group in col_groups:
        if col_name in col_group:
            return(True)
    return(False)


#only add noise to columns not in column pairings
#adds gaussian distribution noise to ints and floats. It maintains the min and max
def add_multiplicative_noise_to_df(df, column_pairing_groups=[]):
    logging.info("Adding noise to numeric columns.....................")

    #int_numerics = ['int16', 'int32', 'int64']
    #float_numerics = ['float16', 'float32', 'float64']
    int_numerics = ['int']
    float_numerics = ['float']

    int_numeric_columns = df.select_dtypes(include=int_numerics).columns.tolist() 
    float_numeric_columns = df.select_dtypes(include=float_numerics).columns.tolist() 

    #among the float columns if the values do not have decimal parts, move them into int columns
    for float_numeric_column in float_numeric_columns:
        num_vec = df[float_numeric_column].tolist()
        if not preprocessor_lib.contains_floats(num_vec):
            int_numeric_columns.append(float_numeric_column)

    #remove all items in the updated int columns from the float columns
    for int_numeric_column in int_numeric_columns:
        if int_numeric_column in float_numeric_columns:
            float_numeric_columns.remove(int_numeric_column)

    numeric_columns = df.select_dtypes(include=int_numerics+float_numerics).columns.tolist() 
    for numeric_column in numeric_columns:
        if is_in_group(numeric_column, column_pairing_groups):
            continue
        min_val = df[numeric_column].min()
        max_val = df[numeric_column].max()
        noise = np.random.normal(1, 0.05, [len(df),])
        if numeric_column in int_numeric_columns:
            df[numeric_column] = (df[numeric_column] * noise).round().astype('Int64')
        else:
            df[numeric_column] = df[numeric_column] * noise
        df.loc[df[numeric_column] > max_val, numeric_column] = max_val
        df.loc[df[numeric_column] < min_val, numeric_column] = min_val

    return


def find_max_distance_for_outliers(embedded_df, nn_nbrs, percentile=0.95):
    distances = []
    for i in range(0, len(embedded_df)):
        neighs = nn_nbrs.kneighbors(embedded_df.iloc[[i]], return_distance=True)
        closest_dist = neighs[0][0][1]
        distances.append(closest_dist)
    dist = np.quantile(distances, percentile)
    #logging.info("{}th percentile of distances: {}".format(percentile, dist))
    return dist
 

# Compute correlations between continuous variables (non-boolean)
def get_column_groups_for_continuous_variables(df, nonboolean_columns, threshold):
    groups = []
    corr = np.abs(df[nonboolean_columns].corr(method='pearson'))
    #corr.to_csv("continuous_corr.csv")
    for i in range(len(nonboolean_columns)):
        for j in range(i+1, len(nonboolean_columns)):
            col1 = nonboolean_columns[i]
            col2 = nonboolean_columns[j]
            if abs(corr.loc[col1, col2]) > threshold:
                if [col1, col2] not in groups and [col2, col1] not in groups:
                    groups.append([col1, col2])
    logging.info("Highly correlated continuous variables: {}".format(groups))
    return(groups)


# Compute correlations between continuous variable and categorical variables
def get_column_groups_between_continuous_and_categorical_variables(df, nonboolean_columns, boolean_columns, threshold):
    groups = []
    for i in range(len(nonboolean_columns)):
        for j in range(len(boolean_columns)):
            col1 = nonboolean_columns[i]
            col2 = boolean_columns[j]
            pbsr_r, pbsr_p = stats.pointbiserialr(df[col1],df[col2])
            #logging.info("continuous variable: {} categorical variable: {} pbsr_r:{}".format(col1, col2, pbsr_r))
            if abs(pbsr_r) > threshold:
                logging.info("Highly correlated continuous variable: {} and categorical variable: {}".format(col1, col2))
                if [col1, col2] not in groups and [col2, col1] not in groups:
                    groups.append([col1, col2])

    logging.info("Highly correlated continuous and categorical variables: {}".format(groups))
    return(groups)


#group same categories together
def get_column_groups_for_same_categories(boolean_columns):
    groups=[]

    for col in boolean_columns:
        if "|" not in col:
            continue

        cat = col.split('|')[0]
        found = False
        while(True):
            for group in groups:
                if col in group:
                    found = True
                    break
                cat_group = [i.split('|')[0] for i in group]
                if cat in cat_group:
                    group.append(col)
                    found = True
            if found:
                break
            else:
                groups.append([col])
                break
    #logging.info("Groups of same name categories")
    #logging.info(groups)
    return(groups)


# bias corrected version of Cramer’s V for association between categorical variables
#(regardless of number of factor levels)
def cramer_v_bias_correct(contingency_tbl):
    chi2,_,_,_ = chi2_contingency(contingency_tbl)
    n = contingency_tbl.values.sum()
    phi2 = chi2/n
    r,k = contingency_tbl.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if phi2corr == min((kcorr-1), (rcorr-1)) == 0:
        test_stat = 0
    else:
        test_stat = (phi2corr / min((kcorr-1), (rcorr-1))) ** (1/2)
    dof = min((kcorr-1), (rcorr-1))
    return(test_stat, dof)


def get_column_groups_for_categorical_variables(df, boolean_columns, threshold):
    groups = []
    for i in range(len(boolean_columns)):
        for j in range(i+1, len(boolean_columns)):
            col1 = boolean_columns[i]
            col2 = boolean_columns[j]
            category_name1 = col1.split('|')
            category_name2 = col2.split('|')
            #if the category names are the same skip them because this has been taken care of
            if col1.startswith(category_name2[0]+'|') and col2.startswith(category_name1[0]+'|'):
                continue

            contingency = pd.crosstab(df[col1], df[col2])

            cram_stat, dof = cramer_v_bias_correct(contingency)
            p = 1 - stats.chi2.cdf(cram_stat, dof)

            #c, p, dof, expected = chi2_contingency(contingency)
            #logging.info("p-value of categorical variables: {} and {} p-value: {}".format(col1, col2, p))
            if p < 0.05:
                if [col1, col2] not in groups and [col2, col1] not in groups:
                    groups.append([col1, col2])

    logging.info("Highly correlated categorical variables: {}".format(groups))
    return(groups)


# Compute correlations between categorical variables (boolean)
def get_column_groups_for_categorical_variables_v2(df, boolean_columns, threshold):
    groups = []
    for i in range(len(boolean_columns)):
        for j in range(i+1, len(boolean_columns)):
            col1 = boolean_columns[i]
            col2 = boolean_columns[j]
            category_name1 = col1.split('|')
            category_name2 = col2.split('|')
            #if the category names are the same skip them because this has been taken care of
            if col1.startswith(category_name2[0]+'|') and col2.startswith(category_name1[0]+'|'):
                continue
            #perform chi-square test
            contingency= pd.crosstab(df[col1], df[col2])
            c, p, dof, expected = chi2_contingency(contingency)
            #logging.info("p-value of categorical variables: {} and {} p-value: {}".format(col1, col2, p))
            if p < 0.05:
                logging.info("Highly correlated categorical variables: {} and {} p-value: {}".format(col1, col2, p))
                if [col1, col2] not in groups and [col2, col1] not in groups:
                    groups.append([col1, col2])
    return(groups)


#threshold is the correlation threshold for the continuous variables
def generate_corr_cols_groups(df, threshold):
    logging.info("Generating correlations between columns...")
    boolean_columns, nonboolean_columns = preprocessor_lib.get_boolean_and_nonboolean_columns(df)

    # Compute correlations between categorical variables (i.e. the columns with boolean values)
    logging.info("Computing correlations between categorical variables (boolean)")
    cat_groups = get_column_groups_for_categorical_variables(df, boolean_columns, threshold)

    # Compute correlations between continuous variables (non-boolean)
    logging.info("Computing correlations between continuous variables (non-boolean)")
    cont_groups = get_column_groups_for_continuous_variables(df, nonboolean_columns, threshold)

    # Compute correlations between continuous variable and categorical variables
    logging.info("Computing correlations between continuous variable and categorical variables")
    cont_cat_groups = get_column_groups_between_continuous_and_categorical_variables(df, nonboolean_columns,
            boolean_columns, threshold)

    #all_groups = cat_name_groups + cat_groups + cont_groups + cont_cat_groups
    all_groups = cat_groups + cont_groups + cont_cat_groups
    return(all_groups)


#takes input in the form of [["a","c","d"],["b","e","f"]]  and generates pairwise column indices
#such as {3:2, 2:1}
def generate_col_pairing_indices(columns, column_pairing_groups):
    logging.info("Generating column pairing indices...")

    col_pairing_indices = {}
    for index in range(len(column_pairing_groups)):
        for col1 in column_pairing_groups[index]:
            bool1_list = (columns.str.startswith(col1+'|') | columns.str.match(col1))
            idx1_list = [i for i, val in enumerate(bool1_list) if val]
            for col1_idx in idx1_list:
                for col2 in column_pairing_groups[index]:
                    bool2_list = (columns.str.startswith(col2+'|') | columns.str.match(col2))
                    idx2_list = [i for i, val in enumerate(bool2_list) if val]
                    for col2_idx in idx2_list:
                        if col2_idx not in col_pairing_indices:
                            if col2_idx>col1_idx:
                                col_pairing_indices[col2_idx] = col1_idx
                        else:
                            if col1_idx < col_pairing_indices[col2_idx]:
                                col_pairing_indices[col2_idx] = col1_idx

    #logging.info("column indices:")
    #for i in range(len(columns)):
    #    logging.info("{} {}".format(i, columns[i]))
    #logging.info('column pairings:')
    #logging.info(col_pairing_indices)
    return(col_pairing_indices)


    
def synthesize(df, method='tsne', metric='euclidean', min_cluster_size=5, max_cluster_size=5, batch_size=1, 
               corr_thresh=0.70, include_outliers=False,
               holdout_cols=[], derived_cols_dict={}, col_pairings=[], imputing_method='simple', index_col='',
               add_noise=False):
    
    #####   inputs
    # df: dataframe where all categorical columns are already converted to numerical values
    # method: method used dimension reduction (options are: 'tsne', 'pca')
    # metric: metric used for 'tsne' only dimension reduction (options are: 'euclidean', 'gower')
    # min_cluster_size: minimum number of parents to use in synthetic point generation
    # max_cluster_size: maximum number of parents to use in synthetic point generation
    # batch_size=1: ratio of the number of synthetic records to the real records. Needs to be an integer
    # corr_thresh=0.70: correlation threshold
    # include_outliers=False whether to use the outliers to generate simulated data
    # holdout_cols: the vector of fixed columns names to omit in dimensionality reduction and to use together
    # derived_cols_dict: the derived keys do not get used when embedding the data (e.g. {'bmi':['height','weight']})
    # col_pairings: a list of groupings to co-segregate (e.g. [['prior_chemo_reg','prior_chemo_time']]
    # imputing_method: the imputing method (options are: 'simple', 'iterative')
    # index_col: index can be a column. It is used to map subjects across different tables

    df.dropna(axis=1, how='all', inplace=True)

    my_df = df.copy()
    n_cols = my_df.shape[1]

    if imputing_method == 'iterative':
        my_df = preprocessor_lib.iterative_impute(my_df)
    else:
        my_df = preprocessor_lib.impute_one_hot_encoded_df(my_df)

    if index_col == '':
        df_to_embed = utilities_lib.drop_columns(my_df, holdout_cols+list(derived_cols_dict.keys()))
    else:
        df_to_embed = utilities_lib.drop_columns(my_df, [index_col]+holdout_cols+list(derived_cols_dict.keys()))

    # Compute correlations
    #corr_cols_groups = generate_corr_cols_groups(df_to_embed, corr_thresh)
    derived_groups = utilities_lib.convert_dict_to_groups(derived_cols_dict)

    #group same categories together
    logging.info("Getting column groups for same categories")
    #boolean_columns, nonboolean_columns = preprocessor_lib.get_boolean_and_nonboolean_columns(df)
    boolean_columns, nonboolean_columns = preprocessor_lib.get_boolean_and_nonboolean_columns(df_to_embed)
    cat_name_groups = get_column_groups_for_same_categories(boolean_columns)

    #column_pairing_groups = col_pairings + corr_cols_groups + derived_groups
    column_pairing_groups = col_pairings + cat_name_groups + derived_groups

    logging.info("column_pairing_groups: {}".format(column_pairing_groups))
    #col_pairing_indices = generate_col_pairing_indices(my_df.columns, column_pairing_groups)
    col_pairing_indices = generate_col_pairing_indices(df_to_embed.columns, column_pairing_groups)

    # Perform tSNE with either gower metric or euclidean metric 
    logging.info("Embedding the data......................")
    if method == 'tsne':
        embedded_df = dimanalysis_lib.reduce_tsne(df_to_embed, n_components=2, metric=metric)
    elif method == 'pca':
        embedded_df = dimanalysis_lib.reduce_pca(df_to_embed, n_components=2)
    #elif method == 'umap':
    #    embedded_df = dimanalysis_lib.reduce_umap(df_to_embed, n_components=2)
    elif method == 'ica':
        embedded_df = dimanalysis_lib.reduce_ica(df_to_embed, n_components=2)
    logging.info("Finished embedding the data.............")
    
    data_size = len(embedded_df)
    max_num_n = max_cluster_size
    if max_cluster_size >= data_size:
        max_num_n = data_size
    min_num_n = min(min_cluster_size, data_size)
    nn_nbrs = NearestNeighbors(n_neighbors=max_num_n).fit(embedded_df)

    max_dist = find_max_distance_for_outliers(embedded_df, nn_nbrs)

    # remove the outliers in the source data
    outlier_indices = []
    if not include_outliers:
        for i in range(0, len(embedded_df)):
            this_index = embedded_df.index[i]
            neighs = nn_nbrs.kneighbors(embedded_df.iloc[[i]], return_distance=True)
            distances = neighs[0][0]
            if distances[1] > max_dist:    #if this is an outlier, skip, do not replicate it
                outlier_indices.append(this_index)
        logging.info("Number of outliers found in original source data: {}".format(len(outlier_indices)))
        #embedded_df_no_outliers = embedded_df.drop(outlier_indices)
        #df_no_outliers = df.drop(outlier_indices)

    sampled = []
    #num_outliers = 0
    step_size = 1
    if batch_size < 1:
        step_size = math.floor(1/batch_size)
        batch_size = 1

    for i in range(0, len(embedded_df)):
        if i in outlier_indices:
            continue
        if i%step_size != 0:
            continue
        #this_index = embedded_df.index[i]
        neighs = nn_nbrs.kneighbors(embedded_df.iloc[[i]], return_distance=True)
        breeding = neighs[1][0]      #[1][0] gives the array of indices of closest neighbors
        #distances = neighs[0][0]
        #if distances[1] > max_dist:    #if this is an outlier, skip, do not replicate it
        #    outlier_indices.append(i)
        #    num_outliers += 1
        #    if not include_outliers:
        #        continue
        
        #TODO: add support for batch_size that are > 1 and not int
        for _ in range(batch_size):
            rows = []
            for col in range(0, n_cols):
                if min_num_n == max_num_n:
                    cluster_size = min_num_n
                else:
                    cluster_size = np.random.randint(min_num_n, max_num_n)
                rand_num = np.random.randint(0, cluster_size)
                #if not include_outliers and distances[rand_num] > max_dist:
                #    rand_num = 0    #if the neighbor index selected is an outlier, pick self
                sample_row_index = breeding[rand_num]
                rows.append(rows[col_pairing_indices[col]] if col in col_pairing_indices else sample_row_index)
            this_row = [int(embedded_df.index[i])]
            for c in range(0, n_cols):
                this_row.append(df.iloc[rows[c]][c])
            sampled.append(this_row)
            
    #logging.info("Number of outliers found: {}".format(num_outliers))

    syn_df = pd.DataFrame(sampled, columns=[index_col]+df.columns.to_list())
    syn_df = syn_df.set_index(index_col)

    #add noise to the numeric columns
    if add_noise: 
        add_multiplicative_noise_to_df(syn_df, column_pairing_groups=column_pairing_groups)
    
    #drop any duplicates from the synthesized set that's also in the original set to secure privacy
    len_before = syn_df.shape[0]
    #duplicated_df = df.merge(syn_df, how = 'inner', indicator=False)
    syn_df = pd.concat([syn_df, df, df]).drop_duplicates(keep=False)
    len_after = syn_df.shape[0]
    logging.info("Found {} duplicates and removed them from the synthetic set.".format(len_before-len_after))
    
    return(syn_df)

