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
import copy
import os
import sys
import time
import logging
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import datetime

import uci_config as config
import preprocessor_lib
import utilities_lib
import analytics_wrapper
import synthesis_wrapper


if __name__ == '__main__':
    print('Loaded the code....................................................')
    start_time = time.time()
    print('Time is %s'%datetime.datetime.now())

    print('Creating directories...............................................')
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if not os.path.exists(config.output_dir+'figs/'):
        os.makedirs(config.output_dir+'figs/')
    print('Finished creating directories......................................')

    print('Starting the logger................................................')
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s [%(levelname)s] %(funcName)s: %(message)s',
                        filename=config.output_dir+config.log_file, filemode='w')
    warnings.filterwarnings("ignore")
    print('Finished starting the logger.......................................')

    file_name = config.data_path+config.data_file
    logging.info(file_name)

    logging.info('Loading the data............................................')
    logging.info('Loading the data............................................')
    df = pd.read_csv(file_name)
    print('Finished loading the data..........................................')

    pdf_page = PdfPages(config.output_dir+config.report_file)

    print('Synthesizing the data..............................................')
    print('Time is %s'%datetime.datetime.now())
    syn_df = synthesis_wrapper.synthesize(df, config)
    print('Finished synthesizing the data.....................................')
    print('Time is %s'%datetime.datetime.now())

    if config.cv_flag:
        logging.info('Running analytics on the real and synthetic data............')
        print('Running analytics on the real and synthetic data...................')
        print('Time is %s'%datetime.datetime.now())
        analytics_wrapper.analyze(df.copy(), syn_df.copy(), config, pdf_page)

    pdf_page.close()
    logging.info('It took {} seconds'.format(time.time() - start_time))
    print('It took {} seconds'.format(time.time() - start_time))

    logging.info('...................Done!....................................')
    print('..........................Done!....................................')

