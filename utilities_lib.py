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
from matplotlib import pyplot as plt
import copy
import math
from functools import reduce
from dateutil.parser import parse
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import logging


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def is_identical(list_a, list_b):
    if len(list_a) != len(list_b):
        return False
    for i in list_a:
        if i not in list_b:
            return False
    return True

def merge_2d_lists(list_a, list_b):
    list_c = list_b.copy()
    for item_a in list_a:
        found = False
        for item_c in list_c:
            if is_identical(item_a, item_c):
                found = True
                break
        if not found:
            list_c.append(item_a)
    return(list_c)

#check if list1 and list2 have any common elements
def if_common_element(list1, list2):
    one = set(list1)
    two = set(list2)
    if (one & two):
        return (True)

    return (False)


#example: dict1={'a':[1,2], 'b':[4]}  and dict2={'a':[6], 'd':[8]}  merged={'a':[1,2,6], 'b':[4], 'd':[8]}
def merge_dictionaries(dict1, dict2):
    merged= { key:dict1.get(key,[])+dict2.get(key,[]) for key in set(list(dict1.keys())+list(dict2.keys())) }
    return(merged)


#converts {"a":["c", "d"],"b":["e","f"]}  to [["a","c","d"],["b","e","f"]]
def convert_dict_to_groups(cols_dict):
    groups = []
    for key in cols_dict.keys():
        this_list=[]
        this_list.append(key)
        for value in cols_dict[key]:
            if value not in this_list:
                this_list.append(value)
        groups.append(this_list)

    return(groups)


def bitwise_or_pair(x1, x2): 
    return(np.bitwise_or(x1, x2))

def bitwise_or_list(x):
    return(reduce(bitwise_or_pair, x))


#quantize the columns with values that have a wide range
def quantize_df(df):
    col_names = list(df)
    for col_name in col_names:
        col = list(df[col_name])
        new_col = quantize_list(col)
        df[col_name] = new_col
    return(df)
     
#quantize a given list of values using a list of quantization levels with step.
def quantize_list(values):
    min_val = math.floor(min(values))
    max_val = math.ceil(max(values))
    step = max(1, int((max_val-min_val)/100.0))
    quantizations = list(range(min_val, max_val, step))
    values = np.array(values)
    quantizations = np.array(quantizations)
    new_values = quantizations[np.argmin(np.abs(np.repeat(values[:, np.newaxis], len(quantizations), axis=1) - quantizations), axis=1)]
    return(new_values)


def get_common_rows(df1, df2):
    common_rows = pd.merge(df1, df2, how='inner')
    return common_rows


# Drop the duplicates between df1 and df2 from df1 and return the modified df1
def drop_duplicates(df1, df2):
    df1 = pd.concat([df1, df2, df2]).drop_duplicates(keep=False)
    return(df1)


#Drop columns starting with names given in columns to accomodate encoding format of column names that start with |
def drop_columns_containing(df, columns):
    for drop_col_name in columns:
        df = df.loc[:, ~df.columns.str.startswith(drop_col_name+'|')]
    for drop_col_name in columns:
        df = df.loc[:, ~df.columns.str.match(drop_col_name)]
    return(df)

def is_date(string, fuzzy=False):
    try:
        parse(string, fuzzy=fuzzy)
        return(True)

    except ValueError:
        return(False)


def get_date_columns(df):
    tmp_df = df.copy()
    tmp_df = tmp_df.select_dtypes(exclude=['int','float'])
    date_cols = []
    for col in tmp_df.columns:
        try:
            tmp_df[col].astype('float')
        except:
            tmp_col = pd.to_datetime(tmp_df[col], errors='coerce')
            tmp_col_na = tmp_col.isna().sum()*100/tmp_col.shape[0]
            if not tmp_col_na>90:
                date_cols +=[col]
    return(date_cols)


def drop_date_columns(df):
    '''Dropping any columns that have dates in them from dataset'''
    date_cols = get_date_columns(df)
    logging.info("dropping all the date columns: {}".format(date_cols))
    df = drop_columns(df, date_cols)
    return(df)


def drop_columns(df, columns):
    '''Drop specified columns from dataset'''
    
    logging.info('Columns deleted: %s'%columns)
    
    for drop_col_name in columns:
        df = df.loc[:, ~df.columns.str.match(drop_col_name)]
    return(df)


def keep_columns_containing(data_df, feature_names):
    original_columns = data_df.columns.to_list()
    all_column_names = []
    for column_name in feature_names:
        all_column_names = all_column_names + [col for col in data_df.columns if column_name in col]

    cols_to_drop = utilities_lib.remove_list_from_list(original_columns, all_column_names)
    new_data_df = data_df.drop(cols_to_drop, axis = 1)
    return(new_data_df)


def remove_item_from_list(ls, val):
    return list(filter(lambda x: x != val, ls))


#removes ls2 from ls1 and returns the results
def remove_list_from_list(ls1, ls2):
    for ls in ls2:
        ls1 = remove_item_from_list(ls1, ls)
    return ls1

