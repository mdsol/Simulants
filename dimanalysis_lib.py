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
import copy
from random import shuffle
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
import gower


def reduce_cca(data_df, n_components=2):
    """
    Uses CCA to reduce dimension.
    Parameters:
    data_df (DataFrame): The input data in DataFrame format
    n_components (int): The number of components to reduce to. Default is 2.
    returns:
    DataFrame: returns the data in the reduced dimension
    """
    new_df = data_df.reset_index(drop=True)
    embedded = CCA(n_components=2).fit_transform(new_df.to_numpy())
    return(pd.DataFrame(embedded, index=data_df.index))


def reduce_ica(data_df, n_components=None):
    """
    Uses ICA to reduce dimension.
    Parameters:
    data_df (DataFrame): The input data in DataFrame format
    n_components (int): The number of components to reduce to. Default is all components.
    returns:
    DataFrame: returns the data in the reduced dimension
    """
    new_df = data_df.reset_index(drop=True)
    embedded = FastICA(n_components=2).fit_transform(new_df)
    return(pd.DataFrame(embedded, index=data_df.index))


def reduce_tsne(data_df, n_components=2, init='pca', metric='euclidean'):
    """
    Uses tSNE to reduce dimension.
    Parameters:
    data_df (DataFrame): The input data in DataFrame format
    n_components (int): The number of components to reduce to. Default is 2.
    returns:
    DataFrame: returns the data in the reduced dimension
    """
    new_df = data_df.reset_index(drop=True)
    if metric == 'gower':
        #tsne = TSNE(n_components=n_components, metric='precomputed', square_distances=True)
        tsne = TSNE(n_components=n_components)
        df_gower = gower.gower_matrix(new_df)
        embedded = tsne.fit_transform(df_gower)
    else:
        #tsne = TSNE(n_components, square_distances=True)
        tsne = TSNE(n_components)
        embedded = tsne.fit_transform(new_df)
        
    return(pd.DataFrame(embedded, index=data_df.index))


def reduce_pca(data_df, n_components=None):
    """
    Uses PCA to reduce dimension.
    Parameters:
    data_df (DataFrame): The input data in DataFrame format
    n_components (float): The number of components or to reduce to. If the number if between 0 and 1, n_components is the % of 
                            the principal components will be kept. Default is all components.
    returns:
    DataFrame: returns the data in the reduced dimension
    """
    new_df = data_df.reset_index(drop=True)
    data_np = new_df.to_numpy()
    
    #Standardize the data by removing the mean and scaling to unit variance
    pca_np = StandardScaler().fit_transform(data_np)
    pca = PCA(n_components)
    embedded = pca.fit_transform(pca_np)
    
    return(pd.DataFrame(embedded, index=data_df.index))

