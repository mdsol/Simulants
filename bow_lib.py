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
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def generate_code_book(data_df, num_bins):
    """
    Generates a BOW codebook for the data.
    Parameters:
    data_df (DataFrame): The input data in DataFrame format
    num_bins (int): The number of BOW bins.
    returns:
    kmeans object: returns the BOW codebook
    """
    x = data_df.values.tolist()
    
    kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(x)
    return kmeans


def get_histogram(kmeans, data_df):
    """
    returns the BOW pdf.
    Parameters:
    kmeans: The kmeans clusters from the codebook generation.
    data_df (DataFrame): The input data in DataFrame format
    returns:
    numpy: returns the pdf
    """
    num_bins = kmeans.cluster_centers_.shape[0]
    centroids = kmeans.cluster_centers_
    data_list = data_df.values.tolist()
    
    nn = NearestNeighbors(n_neighbors=1).fit(centroids)

    histogram = np.zeros(shape=(1, num_bins))
    for j in data_list:
        neighs = nn.kneighbors([j])
        closest_bin_index = neighs[1][0][0]
        histogram[0][closest_bin_index] = histogram[0][closest_bin_index] + 1

    hist = histogram.tolist()[0]
    pdf  = np.divide(hist, sum(hist))

    return pdf

