######################################################################
### Machine Learning - Project 1: Finding the Higgs Boson using Machine Learning ###
######################################################################
Authors:
- Luis da Silva
- Richie Yat Tsai Wan
- Gaëlle Wavre

######################
## Running the program ##
######################
- The data is available in the 'data' folder in the zipped folders "train.csv.zip" and "test.csv.zip"
- To the run the script, go to the corresponding folder in the terminal and type: "python run.py"
- The predictions are generated in the "output" folder

##################
## Folder Structure ##
##################
- 

#############
## Functions ##
#############

## "proj1_helpers.py" ##

- load_csv_data: loads data 
- predict_labels: generates class predictions given weights and a test data matrix
- predict_labels_log: implements the quantize step in logistic regression
- create_csv_submission: creates output file in csv format for submission to kaggle
- cluster_predict: takes clusterized weights, sets and IDs and performs a prediction
- histogram_clusters: visualization of the distribution of all features after having been clustered
- single_histogram: visualization of single histogram with cluster data
- get_feature_names: gets the names of features from the header
- mapping: mapping of index number to and from feature names
- cv_viz: visualization of the curves of MSE of train and test sets


## "implementations.py" ##
- least_squares_GD: linear regression using standard gradient descent
- least_squares_SGD: linear regression using stochastic gradient descent
- least_squares: least squares regression using normal equations
- ridge_regression: ridge regression using normal equations with parameter lambda
- logistic_regression: logistic regression using GD or SGD
- reg_logistic_regression: regularized logistic regression using GD or SGD


## "costs.py" ##

Gradient functions:
- compute_gradient: computes the gradient using the definition of gradient for MSE
- compute_stoch_gradient: computes the stochastic gradient
- compute_log_gradient: computes the gradient for logistic regression

Cost functions:
- compute_mse: computes the loss using mean square error
- compute_mae: computes the loss using mean absolute error
- compute_rmse: computes the root mean square error for ridge and lasso regression
- sigmoid: computes the sigmoid function on input z
- compute_logloss: computes the loss function for logistic regression


## "preprocessing.py" ##

Utility functions:
- batch_iter: generates a minibatch iterator for a dataset
- split_data: splits the dataset according to given split ratio
- sample_data: defines samples from the given dataset

Pre-processing functions:
- standardize: standardizes the dataset to have 0 mean and unit variance
- add_bias: adds a bias at the beginning of the dataset
- build_poly: builds polynomial basis functions for input data up to a given degree
- convert_label: converts the labels into 0 or 1 for logistic regression
- replace_999_nan: replaces all '-999' values by NaN
- replace_999_mean: replaces all '-999' values by the mean of their column
- replace_999_median: replaces all '-999' values by the median of their column
- replace_outliers: replaces outliers that are outside of the given defined confidence interval by the median
- prijetnum_indexing: obtains the indices for the various clusters according to their "PRI_jet_num" value
- prijetnum_clustering: clusters the data into four groups according to their PRI_jet_num value
- delete_features: if the entire column has '-999' values, the data is deleted and the index registered in a list "idx_taken_out" 
- reexpand: rexexpansion of weight vector after computation of weights where some features were deleted

Cluster-processing functions:
- cluster_log: returns the data sets with a natural logarithm applied to selected features
- cluster_std: standardizes the clusterized datasets
- cluster_replace: replaces remaining '-999' values for all sets (by default by the mean)
- cluster_buildpoly: builds polynomial expansion for all clusters with respect to their optimal degree found during crossvalidation
- cluster_preprocessing_train: preprocesses whole training set (clusters them w.r.t PRIjetnum, applies log to wanted features, removes features with all '-999' rows, replaces remaining '-999' with the mean, standardizes, and returns all sets, targets, and deleted column indices)
- cluster_preprocessing_test: identical to "cluster_processing_train" but on the test dataset
- cluster_preprocessing_train_alt: same processing functions on the train dataset but done before the clustering
- cluster_preprocessing_test_alt: same processing functions on the test dataset but done before the clustering

## "train_tune.py" ##
- build_k_indices: builds k indices for k-fold cross validation
- k_split: returns the split datasets for k_fold cross validation
- cross_validation-ridge: returns the losses and weights of ridge regression for a given value of lambda
- crossval_ridge_gridsearch: computes the best degree of polynomial expansion and its associated optimal value of lambda as well as the associated weight that optimizes the RMSE loss for ridge regression using k-fold cross validation
- cross_validation_regulog: returns the losses and weights of regularized logistic regression for a given value of lambda and gamma
- crossval_regulog_gridsearch: computes the best degree of polynomial expansion and its associated optimal value of lambda as well as the associated weight that optimizes the RMSE loss for regularized logistic regression using k-fold cross validation






















