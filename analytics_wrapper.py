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


import pandas as pd
import numpy as np
import logging
from numpy import mean
from numpy import std
from scipy.spatial import distance
from collections import Counter
import matplotlib.pyplot as plt
import math
#from pandas_profiling import ProfileReport

import analytics_lib
import bow_lib
import dimanalysis_lib
import preprocessor_lib
import utilities_lib
import visualization_lib


def analyze_missingness(df):
    missingness_df = pd.DataFrame(columns=['column name', 'percent missing'])
    index = 0
    for column in df.columns:
        missingness = df[column].isna().sum()/len(df[column])
        logging.info('Column %s missingness: %.2f'%(column, missingness))
        if missingness >= 0.3:
            missingness_df.loc[index] = [column, missingness]
            index = index + 1

    return(missingness_df)


def analyze(source_df, syn_df, config, pdf_page):
    source_df = utilities_lib.drop_date_columns(source_df)
    syn_df = utilities_lib.drop_date_columns(syn_df)

    logging.info('Analyzing duplicates in the source data.....................')
    source_duplicates = source_df[source_df.duplicated()]
    dup_len = len(source_duplicates)
    if dup_len > 0:
        logging.warning("Number of duplicates in the source data: {}".format(dup_len))

    logging.info('Analyzing missingness of all the source data................')
    missingness_df = analyze_missingness(source_df)
    missingness_df.to_csv(config.output_dir+config.proj_name+'_missingness_src.csv')

    logging.info('Analyzing missingness of all the synthesized data................')
    analyze_missingness(syn_df)
    missingness_df = analyze_missingness(syn_df)
    missingness_df.to_csv(config.output_dir+config.proj_name+'_missingness_syn.csv')

    source_df = preprocessor_lib.one_hot_encoding_encode(source_df)
    syn_df = preprocessor_lib.one_hot_encoding_encode(syn_df)

    if source_df.shape[1] != syn_df.shape[1]:
        logging.warning("Real and synthesized data do not have the same number of columns")

    logging.info("source_data: {}".format(source_df.shape))
    logging.info("syn_data: {}".format(syn_df.shape))

    source_df.dropna(axis=1, how='all', inplace=True)
    syn_df.dropna(axis=1, how='all', inplace=True)
    #cols_keep = source_df.columns & syn_df.columns
    cols_keep = source_df.columns.intersection(syn_df.columns)

    source_df = source_df[cols_keep]
    syn_df = syn_df[cols_keep]

    source_df = source_df.replace([np.inf,-np.inf,np.nan],0)
    syn_df = syn_df.replace([np.inf,-np.inf,np.nan],0)
    source_df = preprocessor_lib.impute_one_hot_encoded_df(source_df)
    syn_df = preprocessor_lib.impute_one_hot_encoded_df(syn_df)

    pd.set_option('max_columns', None)
    pd.set_option('max_rows', None)
    logging.info("-------data types---------")
    logging.info(source_df.dtypes)

    num_common_rows = len(utilities_lib.get_common_rows(source_df, syn_df))
    if num_common_rows > 0:
        logging.warning("The number of common rows between the real data and the synthesized data is: {}".format(num_common_rows))
    else:
        logging.info("The number of common rows between the real data and the synthesized data is: {}".format(num_common_rows))
    
    logging.info("------Comparing the univariate statistics --------")
    logging.info("------Comparing the mean--------------")
    boolean_columns, nonboolean_columns = preprocessor_lib.get_boolean_and_nonboolean_columns(source_df)
    means = pd.concat([source_df.loc[:, nonboolean_columns].mean(), syn_df.loc[:, nonboolean_columns].mean()], axis=1)
    means.columns = ["Real (mean)", "Synthesized (mean)"]
    logging.info(means)
    means.to_csv(config.output_dir+config.proj_name+'_fid_mean_comparison.csv')
    source_df.mean().to_csv(config.output_dir+config.proj_name+'_fid_mean_real.csv')
    syn_df.mean().to_csv(config.output_dir+config.proj_name+'_fid_mean_syn.csv')

    num_pages = math.ceil(len(means)/20)
    new_means = means.copy().reset_index()
    for i in range(num_pages):
        plt.clf()
        fig, ax = plt.subplots()
        ax.axis('off')
        index_start = i*20
        index_end = index_start + 20
        if len(new_means) != 0:
            the_table = ax.table(cellText=new_means[index_start:index_end].values,colLabels=new_means[index_start:index_end].columns,loc='center')
        pdf_page.savefig(fig)
        plt.clf()

    logging.info("------Comparing the median--------------")
    medians = pd.concat([source_df.loc[:, nonboolean_columns].median(), syn_df.loc[:, nonboolean_columns].median()], axis=1)
    medians.columns = ["Real (median)", "Synthesized (median)"]
    logging.info(medians)
    medians.to_csv(config.output_dir+config.proj_name+'_fid_median_comparison.csv')
    source_df.median().to_csv(config.output_dir+config.proj_name+'_fid_median_real.csv')
    syn_df.median().to_csv(config.output_dir+config.proj_name+'_fid_median_syn.csv')

    logging.info("------Covariance of real data--------------")
    #logging.info(source_df.cov())
    logging.info("------Covariance of synthesized data--------------")
    #logging.info(syn_df.cov())
    source_df.loc[:, nonboolean_columns].cov().to_csv(config.output_dir+config.proj_name+'_fid_cov_real.csv')
    syn_df.loc[:, nonboolean_columns].cov().to_csv(config.output_dir+config.proj_name+'_fid_cov_syn.csv')

    p_thresh = 0.05
    logging.info("Running Fisher Exact and Kolmogorov-Smirnov tests with p_threshold of {} ............".format(p_thresh))
    dissimilar_cols = analytics_lib.extract_dissimilar_features(source_df, syn_df, p_threshold=p_thresh)
    if len(dissimilar_cols) > 0:
        logging.warning("Columns with different distributions between the original and the synthetic data: {}".format(dissimilar_cols))
    else:
        logging.info("Columns with different distributions between the original and the synthetic data: {}".format(dissimilar_cols))


    dissimilar_cols_df = pd.DataFrame(dissimilar_cols, columns=['column name'])
    dissimilar_cols_df.to_csv(config.output_dir+config.proj_name+'_dissimilar_cols.csv', index=False)

    num_pages = math.ceil(len(dissimilar_cols_df)/20)
    for i in range(num_pages):
        plt.clf()
        fig, ax = plt.subplots()
        ax.axis('off')
        index_start = i*20
        index_end = index_start + 20
        if len(dissimilar_cols_df)==0:
            the_table = ax.table(cellText=pd.DataFrame(['']).values,
                        colLabels=['Variables with different distributions from real to synthetic data'],loc='center')
        else:
            the_table = ax.table(cellText=dissimilar_cols_df[index_start:index_end].values,
                        colLabels=['Variables with different distributions from real to synthetic data'],loc='center')
        plt.title("Variables with different distributions from real to synthetic data")
        pdf_page.savefig(fig)
        plt.clf()
   
    logging.info("-----------The p-values-----------------------")
    stats = analytics_lib.compare_columns(source_df, syn_df)
    logging.info(stats.loc[boolean_columns][['p-value', 'odds ratio']])
    logging.info(stats.loc[nonboolean_columns][['p-value', 'ks statistic']])
    
    num_of_rows_source_df = source_df.shape[0]
    logging.info("Reduce dimension of data to 2.........")
    both_sets = pd.concat([source_df, syn_df], axis=0)

    #both_sets_low_dim = dimanalysis_lib.reduce_tsne(both_sets, n_components=2)
    both_sets_low_dim = dimanalysis_lib.reduce_pca(both_sets, n_components=2)
    
    logging.info("Plotting original data.........")
    both_sets_low_dim = both_sets_low_dim.reset_index(drop=True)
    visualization_lib.scatter(both_sets_low_dim[0:num_of_rows_source_df], config.output_dir+'figs/'+config.proj_name+'_real_data.svg', hold=True, pdf_page=pdf_page, c='black', alpha=1)

    logging.info("Plotting synthesized data.........")
    visualization_lib.scatter(both_sets_low_dim[num_of_rows_source_df:-1], config.output_dir+'figs/'+config.proj_name+'_syn_data.svg', pdf_page=pdf_page, c='red', alpha=0.5)

    logging.info("Calculating the silhouette coefficient between the real and the synthetic data.........")
    s = analytics_lib.calculate_silhouette_coef(source_df, syn_df)
    message = 'The silhouette score between real and synthesized data: %.3f'%(s)
    logging.info(message)

    plt.clf()
    fig, ax = plt.subplots()
    ax.axis('off')
    plt.text(0.05,0.95, message, transform=fig.transFigure, size=10)
    pdf_page.savefig(fig)
    plt.clf()

    logging.info("Generating the BOW representation..................")
    codebook = bow_lib.generate_code_book(both_sets_low_dim, config.cv_bow_num_of_bins)
    real_hist = bow_lib.get_histogram(codebook, both_sets_low_dim[0:len(source_df)])
    syn_hist = bow_lib.get_histogram(codebook, both_sets_low_dim[len(source_df):len(both_sets)])

    message = 'Distance between real and synthesized data is: %.3f'%(distance.euclidean(real_hist, syn_hist))
    logging.info(message)
    visualization_lib.bar(real_hist, config.cv_bow_num_of_bins,
            config.output_dir+'figs/'+config.proj_name+'_pca_bow_hist_real.svg', pdf_page=pdf_page, hold=True)
    visualization_lib.bar(syn_hist, config.cv_bow_num_of_bins,
            config.output_dir+'figs/'+config.proj_name+'_pca_bow_hist_syn.svg', pdf_page=pdf_page)

    plt.clf()
    fig, ax = plt.subplots()
    ax.axis('off')
    plt.text(0.05,0.95, message, transform=fig.transFigure, size=10)
    pdf_page.savefig(fig)
    plt.clf()

    logging.info("Plotting the correlation heatmap for the original data...................")
    boolean_columns, nonboolean_columns = preprocessor_lib.get_boolean_and_nonboolean_columns(source_df)
    nonboolean_source_df = source_df.loc[:, nonboolean_columns]
    visualization_lib.correlation_heatmap(nonboolean_source_df, config.output_dir+'figs/'+config.proj_name+'_real_corr.svg', corr='pearson', pdf_page=pdf_page)

    logging.info("Plotting the correlation heatmap for the synthesized data...................")
    boolean_columns, nonboolean_columns = preprocessor_lib.get_boolean_and_nonboolean_columns(syn_df)
    nonboolean_syn_df = syn_df.loc[:, nonboolean_columns]
    visualization_lib.correlation_heatmap(nonboolean_syn_df, config.output_dir+'figs/'+config.proj_name+'_syn_corr.svg', corr='pearson', pdf_page=pdf_page)

    visualization_lib.diff_correlation_heatmap(nonboolean_source_df, nonboolean_syn_df, config.output_dir+'figs/'+config.proj_name+'_diff_corr.svg', corr='pearson', pdf_page=pdf_page)


    return

