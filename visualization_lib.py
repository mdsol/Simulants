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
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def scatter_by_column_names(df, x_name, y_name, filename, hold=False):
    """
    Create a scatter plot and save to a file.
    Parameters:
    df (DataFrame): The input data in DataFrame format
    x_name (list): Column names for X.
    y_name (list): Column names for Y.
    filename (str): The filename to save the plot to.
    """
    x = (df.loc[:,x_name].values.tolist())
    y = (df.loc[:,y_name].values.tolist())
    plt.scatter(x, y)
    #plt.show()
    plt.savefig(filename)
    if (not hold):
        plt.clf()
    return

def scatter(df, filename, hold=False, pdf_page=None, label=None,c='black',alpha=0.5):
    """
    Create a scatter plot of the first two columns and save to a file.
    Parameters:
    df (DataFrame): The input data in DataFrame format
    filename (str): The filename to save the plot to.
    """
    x = (df.iloc[:,0].values.tolist())
    y = (df.iloc[:,1].values.tolist())
    plt.scatter(x, y, label=label,c=c, alpha=alpha, edgecolor='white')
    #plt.show()
    plt.legend(bbox_to_anchor=(1.0, 0.6))
    plt.title(filename)
    plt.tight_layout()
    fig = plt.savefig(filename)
    if not hold and pdf_page != None:
        pdf_page.savefig(fig, bbox_inches='tight')
    if (not hold):
        plt.clf()
    return


def scatter_bw(df, filename, hold=False, pdf_page=None):
    """
    Create a scatter plot of the first two columns and save to a file.
    Parameters:
    df (DataFrame): The input data in DataFrame format
    filename (str): The filename to save the plot to.
    """
    x = (df.iloc[:,0].values.tolist())
    y = (df.iloc[:,1].values.tolist())

    if hold:
        color = ['0' for item in y]
    else:
        color = ['0.8' for item in y]

    plt.scatter(x, y, c=color, alpha=0.5, edgecolor='white')
    plt.title(filename)
    fig = plt.savefig(filename)
    if not hold and pdf_page != None:
        pdf_page.savefig(fig, bbox_inches='tight')
    if (not hold):
        plt.clf()
    return


def histogram(df, bins, y, filename):
    fig = df.plot(bins, y, kind='hist', alpha=0.5)
    #df.show()
    fig.savefig(filename).get_figure()
    return

def bar(data_np, num_of_bins, filename, pdf_page=None, hold=False):
    bins = list(range(0, num_of_bins))
    plt.bar(bins, data_np, alpha=0.5, edgecolor='white')
    plt.title(filename)
    #plt.show()
    fig = plt.savefig(filename)
    if pdf_page != None:
        pdf_page.savefig(fig, bbox_inches='tight')
    if not hold:
        plt.clf()
    
def correlation_heatmap(df, filename, corr='pearson', pdf_page=None):
    #df_corr = np.abs(df.corr(method=corr))
    df_corr = (df.corr(method=corr))
    #order the column and row names alphabetically
    df_corr = df_corr.reindex(sorted(df_corr.columns), axis=0)
    df_corr = df_corr.reindex(sorted(df_corr.columns), axis=1)
    #logging.info(df_corr)
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(filename)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    #sns_plot = sns.heatmap(df_corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    sns_plot = sns.heatmap(df_corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    fig = sns_plot.get_figure()
    fig.savefig(filename)
    if pdf_page != None:
        pdf_page.savefig(fig, bbox_inches='tight')
    fig.clf()
    
# Plots the heatmap of diffrence of correlation matrices of two dataframes
def diff_correlation_heatmap(df1, df2, filename, corr='pearson', pdf_page=None):
    #df1_corr = np.abs(df1.corr(method=corr))
    df1_corr = (df1.corr(method=corr))
    #order the column and row names alphabetically
    df1_corr = df1_corr.reindex(sorted(df1_corr.columns), axis=0)
    df1_corr = df1_corr.reindex(sorted(df1_corr.columns), axis=1)
    #logging.info(df1_corr)

    #df2_corr = np.abs(df2.corr(method=corr))
    df2_corr = (df2.corr(method=corr))
    #order the column and row names alphabetically
    df2_corr = df2_corr.reindex(sorted(df2_corr.columns), axis=0)
    df2_corr = df2_corr.reindex(sorted(df2_corr.columns), axis=1)
    #logging.info(df2_corr)
    
    df_corr = (abs(df1_corr.fillna(0)-df2_corr.fillna(0))).fillna(0)
    print("The sum of diff_corr: {}".format(df_corr.values.sum()))
    logging.info("The sum of diff_corr: {}".format(df_corr.values.sum()))
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    #sns_plot = sns.heatmap(df_corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    sns_plot = sns.heatmap(df_corr, mask=mask, cmap=cmap, vmin=-2, vmax=2, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title(filename)
    fig = sns_plot.get_figure()
    fig.savefig(filename)
    if pdf_page != None:
        pdf_page.savefig(fig, bbox_inches='tight')
    fig.clf()
