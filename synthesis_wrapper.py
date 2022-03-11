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


import logging

import k_anonymity
import synthesis_lib
import preprocessor_lib
import utilities_lib


def synthesize(df, config):

    logging.info('Performing k-anonymity to the data......................')
    logging.info('The data size before k-anonymity: {}'.format(df.shape))
    ignore_columns = utilities_lib.get_date_columns(df)
    df = k_anonymity.perform_k_anonymity(df, config.anonymity_k, ignore_columns)
    logging.info('The data size after k-anonymity: {}'.format(df.shape))

    ignore_columns = utilities_lib.get_date_columns(df)
    tmp_df = df.loc[:, ~df.columns.isin(ignore_columns)]
    label_encoded_df, encoding_dict = preprocessor_lib.label_encoding_encode(tmp_df)
    label_encoded_df = preprocessor_lib.impute_label_encoded_df(label_encoded_df)

    corr_cols_groups = synthesis_lib.generate_corr_cols_groups(label_encoded_df, config.corr_threshold)
    col_pairings = utilities_lib.merge_2d_lists(corr_cols_groups, config.col_pairings)

    one_hot_encoded_df = preprocessor_lib.one_hot_encoding_encode(tmp_df)
    logging.info("encoded_df: {}".format(one_hot_encoded_df.shape))

    encoded_df = one_hot_encoded_df

    logging.info('Synthesizing the data data.............................')

    syn_encoded_df = synthesis_lib.synthesize(encoded_df,
            method=config.embedding_method, metric=config.embedding_metric,
            min_cluster_size=config.min_cluster_size, max_cluster_size=config.max_cluster_size,
            batch_size=config.batch_size, corr_thresh=config.corr_threshold, include_outliers=config.include_outliers,
            holdout_cols=config.holdout_cols, derived_cols_dict={}, col_pairings=col_pairings,
            imputing_method=config.imputing_method, add_noise=config.add_noise)
    logging.info("syn_encoded_df: {}".format(syn_encoded_df.shape))

    logging.info('Decoding the synthesized data...............................')
    syn_encoded_df_no_index = syn_encoded_df.reset_index(drop=False)
    syn_df = preprocessor_lib.one_hot_encoding_decode(syn_encoded_df_no_index)

    logging.info('Saving the synthesized data.....................................')
    logging.info('syn_df: {}'.format(syn_df.shape))

    df = df.reset_index(drop=False)

    df_columns = utilities_lib.intersection(df.columns, syn_df.columns)
    syn_df = syn_df.reindex(columns=df_columns)
    syn_df.to_csv(config.output_dir+config.proj_name+'_syn.csv', index=False)

    return(syn_df)


