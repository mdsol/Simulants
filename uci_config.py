###############################################################################
# General config parameters
###############################################################################
proj_name = 'demo'                     # project name
data_path='./uci-heart-disease/'       # directory containing the source data
data_file='processed.cleveland.csv'    # file name for the source data in csv format 
output_dir='./output_'+proj_name+'/'   # output directory where the synthesized data will be placed
log_file=proj_name+'.log'              # name of the log file
report_file=proj_name+'_report.pdf'    # name of the report file in pdf containing the cross-validations
num_cpus=1                             # number of CPUs to use


###############################################################################
# Core Simulants synthesizer config parameters
###############################################################################
anonymity_k = 1                        # k-anonymity for the categorical attributes
embedding_method = 'tsne'              # method for embedding; options: cca, ica, tsne, pca
embedding_metric = 'gower'             # metric to use for tsne; options: gower, euclidean
min_cluster_size = 5                   # minimum cluster size for knn
max_cluster_size = 5                   # maximum cluster size for knn
corr_threshold = 0.7                   # correlation coefficient threshold for co-segregation of attributes
batch_size = 1                         # ratio of the number of synthesized data to source data
include_outliers = True                # whether to include outlisers in the synthesized data
col_pairings = []                      # columns that need to be forced to be co-segregated.
                                       # example: [['age', 'weight'], ['ethnicity', 'race']]]
holdout_cols = []                      # name of columns to holdout before embedding is done
imputing_method = 'simple'             # imputation method to use before embedding; options: simple, iterative
add_noise = True                       # whether to add gaussian noise to the numerical attributes


###############################################################################
# Fidelity (cross-validation) config parameters
###############################################################################
cv_flag = True                         # flag to perform cross-validation
cv_bow_num_of_bins = 40                # number of bins to use for the bag-of-words cross-validation

