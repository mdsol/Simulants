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
from sas7bdat import SAS7BDAT
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import math


#checks whether a given float contains decimals
def contains_decimal(x):
    frac, whole = math.modf(x)
    if frac>0:
        return(True)
    return(False)


#checks weather the given vector of numbers contains floats, otherwise all are assumed to be ints
def contains_floats(x_vec):
    highest_prec = False
    for x in x_vec:
        prec = contains_decimal(x)
        if prec:
            highest_prec = prec
    return(highest_prec)


def get_boolean_and_nonboolean_columns(df):
    boolean_columns = []
    for c in df.columns:
        if set(df.loc[:,c]).issubset(set([0, 1])) or '|' in c:
            boolean_columns.append(c)
    nonboolean_columns = list(set(df.columns) - set(boolean_columns))
    return(boolean_columns, nonboolean_columns)


#LabelEncoding: converts a dataframe containing categorical values to numbers and returns the new dataframe as well as the dictionary of the mappings
def label_encoding_encode(df):
    #df = pd.DataFrame({'value': ["a", "b", "c", "a"], 'num': [1,2,3,4], 'name':["yy","bb", "yy","zz"]})
    df_encoded = df.copy()
    df_type = df.dtypes
    object_idx = np.where(df_type == 'object')
    dicts = {}
    for i in range(0, len(object_idx[0])):
        c = df_encoded[df.columns[object_idx[0][i]]].astype('category')
        d = dict(enumerate(c.cat.categories))
        col_name = df.columns[object_idx[0][i]]
        df_encoded[col_name] = c.cat.codes
        dicts[col_name] = d
    
    return (df_encoded, dicts)


#LabelEncoding: converts a dataframe containing numbers to categorical values using a mapping dictionary and returns a new dataframe
def label_encoding_decode(df_encoded, dicts):
    df = df_encoded.copy()
    keys=dicts.keys()
    for key in keys:
        df[key] = df_encoded[key].map(dicts[key])
    
    return (df)


def one_hot_encoding_encode(df):
    encoded_df = pd.get_dummies(df, prefix_sep='|')
    return(encoded_df)


def one_hot_encoding_decode(df_dummies):
    prefix_sep = '|'
    pos = defaultdict(list)
    vals = defaultdict(list)

    for i, c in enumerate(df_dummies.columns):
        if prefix_sep in c:
            k, v = c.split(prefix_sep, 1)
            pos[k].append(i)
            vals[k].append(v)
        else:
            pos[prefix_sep].append(i)

    df = pd.DataFrame({k: pd.Categorical.from_codes(
                              np.argmax(df_dummies.iloc[:, pos[k]].values, axis=1),
                              vals[k])
                      for k in vals})

    df[df_dummies.columns[pos[prefix_sep]]] = df_dummies.iloc[:, pos[prefix_sep]]
    return df


# This function takes a dataframe with categorical and ordinal columns and converts all fields to floats
# It performs one-hot-encoding for the categorical variables and label encoding for the ordinal variables
def encode_df(df, categorical_columns, ordinal_columns):
    all_columns = df.columns.tolist()
    all_columns_set = set(all_columns)
    other_columns_set = all_columns_set.difference(set(categorical_columns))
    other_columns = list(rest_columns_set.difference(set(ordinal_columns)))
    
    cat_df = df[categorical_columns]
    ord_df = df[ordinal_columns]
    other_df = df[other_columns]
    
    cat_df_encoded = one_hot_encoding_encode(cat_df)
    ord_df_encoded, ord_dict = label_encoding_encode(ord_df)
    
    encoded_df = other_df.join(cat_df_encoded)
    encoded_df = encoded_df.join(ord_df_encoded)
    
    return(encoded_df, ord_dict)
    
    
# This function converts back an encoded dataframe to the original categorical and ordinal columns
def decode_df(df, ord_dict):
    decoded_df = label_encoding_decode(df, ord_dict)
    decoded_df = one_hot_encoding_decode(decoded_df)
    return(decoded_df)


#fill the missing data with new values not existing in the column
#this is used to determine the column correlations
def impute_label_encoded_df(df):
    for column in df.columns:
        tmp_col_values = sorted(df[column].unique())
        col_values = [x for x in tmp_col_values if math.isnan(x) == False]
        if len(col_values) >= 2:
            fill_val = col_values[len(col_values)-1]+(col_values[len(col_values)-1]-col_values[len(col_values)-2])
        elif len(col_values) == 1:
            if col_values[0] != 0:
                fill_val = 2*col_values[0]
            else:
                fill_val = 1
        else:
            fill_val = 0
        df[[column]] = df[[column]].fillna(value=fill_val)

    return(df)


#impute the missing values of boolean columns with the most frequent value and
#impute the missing values of the non-boolean columns with the median
def impute_one_hot_encoded_df(df):
    boolean_columns, nonboolean_columns = get_boolean_and_nonboolean_columns(df)
    boolean_df = df[boolean_columns]
    nonboolean_df = df[nonboolean_columns]

    if len(nonboolean_columns) != 0:
        imputed_nonboolean_df = nonboolean_df.fillna(nonboolean_df.median())
        imputed_nonboolean_df = imputed_nonboolean_df.reset_index(drop=True)

    if len(boolean_columns) != 0:
        imp_most_freq = SimpleImputer(strategy='most_frequent')
        imp_most_freq.fit(boolean_df)
        imputed_boolean_df = pd.DataFrame(imp_most_freq.transform(boolean_df))
        imputed_boolean_df.columns = boolean_df.columns
        imputed_boolean_df = imputed_boolean_df.reset_index(drop=True)

    if len(nonboolean_columns) != 0 and len(boolean_columns) != 0:
        imputed_df = pd.concat([imputed_nonboolean_df, imputed_boolean_df], axis=1)
    elif len(nonboolean_columns) == 0:
        imputed_df = imputed_boolean_df
    elif len(boolean_columns) == 0:
        imputed_df = imputed_nonboolean_df

    imputed_df = imputed_df.reindex(columns=df.columns)
    imputed_df = imputed_df.set_index(df.index)
    return(imputed_df)


def iterative_impute(df):
    logging.info("Iterative imputing the data")
    imputer = IterativeImputer()
    imputer.fit(df)
    imputed_np = imputer.transform(df)
    imputed_df = pd.DataFrame(imputed_np, columns=df.columns)
    imputed_df = imputed_df.reindex(columns=df.columns)
    imputed_df = imputed_df.set_index(df.index)
    return(imputed_df)
