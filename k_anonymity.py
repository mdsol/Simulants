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
import logging


#drop all rows with any column having less than k distinct values
def perform_k_anonymity(df, anonymity_k, ignore_columns):

    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(df.columns) - set(num_cols))
    for ignore_column in ignore_columns:
        if ignore_column in cat_cols:
            cat_cols.remove(ignore_column)

    for column in cat_cols:
        removals = df[column].value_counts().reset_index()
        removals = removals[removals[column] >= anonymity_k]['index'].values
        df = df[df[column].isin(removals)]

    return(df)


