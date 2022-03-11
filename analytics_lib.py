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



import os
import sys
import copy
from random import shuffle
import pandas as pd
import numpy as np
import scipy.stats as stats
import logging

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, pairwise_distances, accuracy_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

import preprocessor_lib
import utilities_lib


def calculate_silhouette_coef(x1, x2):
    x_all = pd.concat([x1, x2])
    
    y1 = pd.DataFrame(len(x1)*[1])
    y2 = pd.DataFrame(len(x2)*[2])
    y_all = pd.concat([y1, y2])
    
    s_1_2 = silhouette_score(x_all, np.ravel(y_all), metric='euclidean')
    
    return(s_1_2)


def remove_correlated_variables(data_df, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = abs(data_df.corr())
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in data_df.columns:
                    del data_df[colname] # deleting the column from the dataset data_df
    return data_df

def train_two_class_classifier(x_df, y_df, classifier_function=RandomForestClassifier, n_splits=10, n_repeats=3):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    model = classifier_function()
    n_aucs = cross_val_score(model, x_df.to_numpy(), y_df.to_numpy().ravel(), scoring='roc_auc', cv=cv, n_jobs=-1, error_score='raise')
    return(np.mean(n_aucs), np.std(n_aucs))


def train_two_class_classifier_single_fold(x_df, y_df, classifier_function=RandomForestClassifier, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=test_size)
    clf = classifier_function(max_iter=1000, multi_class='auto').fit(x_train, np.ravel(y_train))
    y_pred = clf.predict_proba(x_test)
    auc = roc_auc_score(y_test, y_pred[:,1])
    return(auc)


def train_multiclass_classifier(x_df, y_df, classifier_function=RandomForestClassifier, n_splits=10, n_repeats=3):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    model = classifier_function()
    n_scores = cross_val_score(model, x_df.to_numpy(), y_df.to_numpy().ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return(np.mean(n_scores), np.std(n_scores))


def train_multiclass_classifier_single_fold(x_df, y_df, classifier_function=RandomForestClassifier, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=test_size)
    clf = classifier_function(max_iter=1000, multi_class='auto').fit(x_train, np.ravel(y_train))
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return(acc)


def compare_columns(df1, df2):
    
    if len(df1) != len(df2):
        logging.warning("The lengths of the columns are not the same")
        
    boolean_variables, nonboolean_variables = preprocessor_lib.get_boolean_and_nonboolean_columns(df1)

    stats_df = pd.DataFrame(index = set(boolean_variables) or set(nonboolean_variables),
                            columns=['p-value','odds ratio','ks statistic','method'])
    
    for c in boolean_variables: 
        contingency_table = [[(df1.loc[:,c]==0).sum(),(df2.loc[:,c]==0).sum()], [(df1.loc[:,c]==1).sum(),(df2.loc[:,c]==1).sum()]]
        
        #high p-value means we don't reject the null hypothesis
        stats_df.loc[c,'odds ratio'], stats_df.loc[c,'p-value'] = stats.fisher_exact(contingency_table)
        stats_df.loc[c,'method'] = 'Fisher Exact Test'

    for c in nonboolean_variables:
        #high p-value means they come from the same distribution
        stats_df.loc[c,'ks statistic'], stats_df.loc[c,'p-value'] = stats.ks_2samp(df1.loc[:,c], df2.loc[:,c])
        stats_df.loc[c,'method'] = 'Kolmogorov-Smirnov test'
    
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    
    return(stats_df)

# Kolmogorov-Smirnov Test: A P-value of less than 0.05 is considered significant. 
# The P-value less than significance level rejects the null hypothesis as we expect 
# to see the observed outcome only 5% of the time if the null hypothesis was true.
def extract_dissimilar_features(x1, x2, p_threshold=0.05):
    
    #logging.info("Comparing dataset 1 and dataset 2")
    num_rows = min(len(x1), len(x2))
    stats_df = compare_columns(x1[0:num_rows], x2[0:num_rows])
    
    d = stats_df.loc[(stats_df['p-value'] < p_threshold) & (stats_df['method'] == 'Fisher Exact Test')]
    #cols1 = [utilities_lib.get_feature_name_v2(col) for col in d.index.tolist()]
    cols1 = [col for col in d.index.tolist()]
    d = stats_df.loc[(stats_df['p-value'] < p_threshold) & (stats_df['method'] == 'Kolmogorov-Smirnov test')]
    #cols2 = [utilities_lib.get_feature_name_v2(col) for col in d.index.tolist()]
    cols2 = [col for col in d.index.tolist()]
    cols_1_2 = set(cols1+cols2)
    #logging.info(cols_1_2)
    
    return(list(cols_1_2))
    

def select_features(x1, x2, threshold=0.008):
    y1 = pd.DataFrame(len(x1)*[1])
    y2 = pd.DataFrame(len(x2)*[2])
    x_all = pd.concat([x1, x2])
    y_all = pd.concat([y1, y2])
    
    # Build a forest and compute the impurity-based feature importances
    #forest = RandomForestClassifier()
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

    forest.fit(x_all, np.ravel(y_all))
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    logging.info("Feature ranking:")

    significant_features = []
    nonsignificant_features = []
    for f in range(x_all.shape[1]):
        index = indices[f]
        column_name = x_all.columns[index]
        feature_name = utilities_lib.get_feature_name_v2(column_name)
        
        if importances[indices[f]] > threshold:
            logging.info("%d. feature %s (%f)" % (f + 1, column_name, importances[indices[f]]))
            if feature_name not in significant_features:
                significant_features.append(feature_name)
        else:
            if feature_name not in nonsignificant_features:
                nonsignificant_features.append(feature_name)
    
    return(significant_features, nonsignificant_features)


def preprocess_km(df, event_col='DEATH_FLAG', event_val='True', time_to_event_col='TTE_DEATH', time_to_censor_col='TTE_DEATH'):
    encoded_event_col = event_col + '|' + event_val
    if time_to_censor_col is not None:
        df['survival_drv'] = (utilities_lib.ifelse(df[encoded_event_col].isin([1.0]), 
                                                    df[time_to_event_col], 
                                                    df[time_to_censor_col])).values
    else:
        df['survival_drv'] = df[time_to_event_col].values
    
    df = df.loc[df['survival_drv'] >= 0, :].reset_index(drop=True)
    
    df['survival_event'] = pd.Series([0 for x in range(len(df.index))], index=df.index)
    df.loc[df[encoded_event_col].isin([1.0]), 'survival_event'] = 1
    df.loc[~(df[encoded_event_col].isin([1.0])), 'survival_event'] = 0
    return(df)


def km_wrapper(df, event_col='survival_event', time_to_event_col='survival_drv', cohort_col = None, tick_interval = 200):
    # check for duplicates in subjects
    km_df = df.copy()
    
    # fit to each cohort
    if cohort_col is None:
        km_df['cohort'] = np.zeros(km_df.shape[0])
        cohort_col = 'cohort'
    
    out_tables = {}
    out_models = {}
    grps = km_df[cohort_col].unique()
    at_risk_summary = pd.DataFrame()
    for i in range(len(grps)):
        kmf = KaplanMeierFitter()
        T = km_df.loc[km_df[cohort_col] == grps[i], time_to_event_col]
        C = km_df.loc[km_df[cohort_col] == grps[i], event_col]
        kmf.fit(T, C, label = grps[i])
        
        tmp_tbl = kmf.event_table
        tmp_tbl = pd.merge(tmp_tbl,
                           kmf.survival_function_.rename(columns = {str(grps[i]) : 'survival_prob'}),
                           left_index = True, right_index = True)
        tmp_tbl = pd.merge(tmp_tbl, kmf.confidence_interval_.rename(columns = {str(grps[i]) + '_lower_0.95' : 'ci_lower',
                                                                               str(grps[i]) + '_upper_0.95' : 'ci_upper'}),
                           left_index = True, right_index = True)
        
        # get at risk count at tick intervals
        missing_intervals = {i for i in range(0, int(max(tmp_tbl.index)+1), tick_interval)}.difference(tmp_tbl.index)
        try:
            max_val = max(missing_intervals)
        except:
            max_val = 0
        missing_intervals.add(max_val + tick_interval)
        at_risk_tbl = pd.merge(tmp_tbl['at_risk'],
                               pd.DataFrame(index = missing_intervals),
                               how = 'outer', left_index = True, right_index = True)
        at_risk_tbl = at_risk_tbl.sort_index(axis=0).fillna(method='bfill').rename(columns = {'at_risk' : grps[i]})
        at_risk_summary = pd.concat([at_risk_summary, at_risk_tbl.loc[{i for i in range(0, int(max(at_risk_tbl.index)+1),
                                    tick_interval)},:].sort_index(axis=0).transpose()], axis = 0).fillna(0)
        
        out_tables[grps[i]] = tmp_tbl
        
        out_models[grps[i]] = {'fitted_km' : kmf, 'label' : grps[i]}
    
    return(T, C, out_tables, out_models, at_risk_summary)


def plot_km(df, event_flag, event_value, time_to_event_col, time_to_censor_col, label_text, c='black'):
    df_km = preprocess_km(df, event_col=event_flag, event_val=event_value,
            time_to_event_col=time_to_event_col, time_to_censor_col=time_to_censor_col)
    T, C, km_tables, km_fit, at_risk_table = km_wrapper(df_km)
    plot = km_fit[0.0]['fitted_km'].plot(label=label_text, color=c)
    #plot.set_xlim(0,5000)
    plot.set_xlim(0,1000)
    plot.set_ylim(0,1.0)
    fig = plot.get_figure()
    return(fig)

def km_log_rank(src_df, syn_df, event_flag, event_value, time_to_event_col, time_to_censor_col):
    src_df_km = preprocess_km(src_df, event_col=event_flag, event_val=event_value,
            time_to_event_col=time_to_event_col, time_to_censor_col=time_to_censor_col)
    src_T, src_C, src_km_tables, src_km_fit, src_at_risk_table = km_wrapper(src_df_km)

    syn_df_km = preprocess_km(syn_df, event_col=event_flag, event_val=event_value,
            time_to_event_col=time_to_event_col, time_to_censor_col=time_to_censor_col)
    syn_T, syn_C, syn_km_tables, syn_km_fit, syn_at_risk_table = km_wrapper(syn_df_km)

    lr_summary = logrank_test(src_T, syn_T, src_C, syn_C)#, alpha=99)
    lr_summary.print_summary()
    #print("p_value={} test_statistic={}".format(lr_summary.p_value, lr_summary.test_statistic))
    logging.info("p_value={} test_statistic={}".format(lr_summary.p_value, lr_summary.test_statistic))

    return

