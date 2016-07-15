
# coding: utf-8

# In[20]:

import pandas as pd
import numpy as np

from pandas.io.data import get_data_yahoo
from datetime import datetime
import datetime as dt

def predictStock(company, forcast = 5, endTime = None, startTime = None, debug=False):
	"""
	Predict [company]'s stock trend in [forcast] days by [endTime]
	Returns RISE/FALL trend and confidence
	"""

	if startTime == None:
		startTime = datetime(2000,1,1)
	
	if endTime == None:		
		endTime = dt.date.today() # cuz it's dt's attribute


	# In[21]:

	# Introduce stock indices: nasdaq,dji,frankfurt,london,tokyo,hk,australia
	def getStockData(symbol, start, end, getVolume=False):
	    """
	    Downloads Stock from Yahoo Finance.
	    Returns daily Adj Close and Volume.
	    Returns pandas dataframe.
	    """
	    print "retriving data of", symbol

	    df = get_data_yahoo(symbol, start, end)

	    print "done"

	    df.rename(columns = {'Adj Close' : 'AdjClose' + '_' + symbol}, inplace=True)
	    
	    ed = -2 if getVolume else -1
	    return df.drop(df.columns[:ed], axis = 1)


	# In[22]:

	## Introduce indices:
	# NASDAQ Composite (^IXIC Yahoo Finance)
	# Dow Jones Industrial Average (^DJI Quandl)
	# Frankfurt DAX (^GDAXI Yahoo Finance)
	# London FTSE-100 (^FTSE Yahoo Finance)
	# Paris CAC 40 (^FCHI Yahoo Finance)
	# Tokyo Nikkei-225 (^N225 Yahoo Finance)
	# Hong Kong Hang Seng (^HSI Yahoo Finance)
	# Australia ASX-200 (^AXJO Yahoo Finance)
	## -----------------------------------------------

	indices_list = [company,'^IXIC','^DJI','^GSPC','^GDAXI','^FTSE','^N225','^HSI','^AXJO']
	raw_data = []
	raw_data.append(getStockData(indices_list[0], startTime, endTime, getVolume=True))
	volume_data = raw_data[0].Volume
	raw_data[0] = raw_data[0].drop(raw_data[0].columns[0], axis = 1)

	for indice in indices_list[1:]:
	    raw_data.append(getStockData(indice, startTime, endTime, getVolume=False))

	if debug:
		print '\n\n\n================GENERATE FEATURE DATA==============================\n'
		print 'no. of indicators(prices/indices) retrived: ', len(raw_data)
		print '\n A glimpse of retrived data: \n', raw_data[0].head(), '\n', raw_data[0].tail()


	# In[23]:

	## define parameters
	## -----------------------------------------------

	N_history = 13	# look back N_history * forcast days' data to design features
	return_eval_list = [1,2,3,5,8,13]	# calculate return w.r.t the period listed in this list
	avg_return_span_list = return_eval_list[1:]	 # calculate avg return across the period listed in this list
	volume_rolling_win = N_history * forcast	# volume statistics is evaluated by this rolling window size
	PCA_n = 10	# no. of features for PCA 

	## merge stock prices and indices and generate past N_history day records as features
	## -----------------------------------------------

	meg_data = pd.concat(raw_data, axis=1, join='inner')
	#print meg_data.head()
	for i in range(len(raw_data)):
	    for j in range(1, 10):
	        t = forcast * j;
	        meg_data[meg_data.columns.values[i] + str(t)] = meg_data.iloc[:, i].shift(t)

	if debug:
		print "\n Merged Data with past N_history day records as features: \n", meg_data.tail()

	## Generating return and average return over a list of period to capture stock history movements
	## -----------------------------------------------

	returnList = forcast * np.array(return_eval_list)
	avgReturnList = forcast * np.array(avg_return_span_list)
	
	if debug:
		print "\n return calculation period list \n",returnList
		print "\n avg. return calculation period list \n", avgReturnList

	return_data = pd.DataFrame()
	return_data['return'] = meg_data.iloc[:,0].pct_change(returnList[0])

	for i in returnList[1:]:
	    return_data['return'+str(i)] = meg_data.iloc[:,0].pct_change(i)

	for i in avgReturnList:
	    return_data['avgReturn'+str(i)] = return_data['return'].rolling(i).mean()
	    
	if debug:
		print "\n return_data \n", return_data.tail()

	## Normalize Volume Data and compare to its standard deviation
	## -----------------------------------------------

	volume_norm = (volume_data - volume_data.rolling(volume_rolling_win).mean()) / volume_data.rolling(volume_rolling_win).std()
	
	if debug:
		print "\n rolling normallized volume \n", volume_norm.tail()


	# In[24]:

	import renders as rs
	from IPython.display import display
	from sklearn.decomposition import PCA


	## PCA
	## -----------------------------------------------

	# TODO: Scale the data using the natural logarithm
	meg_data = meg_data.dropna()
	log_data = np.log(meg_data)
	#print "log data: ", log_data

	# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
	pca = PCA(n_components = PCA_n)
	pca.fit(log_data)
	
	if debug:
		print '\n\n\n================PCA==============================\n'
		print "PCA explained variance ratio = ", pca.explained_variance_ratio_

		# Generate PCA results plot
		pca_results = rs.pca_results(log_data, pca)

	# TODO: Transform the data using the PCA fit above
	reduced_data = pca.transform(log_data)
	reduced_data = pd.DataFrame(reduced_data, index=log_data.index)
	#print reduced_data.shape, reduced_data

	## Combine Overall Data Set
	## -----------------------------------------------

	data = pd.concat([reduced_data, return_data, volume_norm], axis=1, join='inner')
	data = data.dropna()
	#print data

	## Generate Labels
	## -----------------------------------------------

	y_raw = return_data['return'].shift(-returnList[0])
	comp = lambda x: (1 if x > 0 else -1)
	y_raw = y_raw.apply(comp)
	if debug:
		print "\n y_raw: \n", y_raw

	x_endtime = data.iloc[-1]    # feature of last day, used for final stock prediction
	data = data.drop(data.index[-returnList[0]:], axis = 0)    # remove last forcast days, useless for model training
	y = pd.Series(y_raw, index = data.index)    # align labels to data indexes

	if debug:
		print "\n data for train/test: ", data.shape, "\n", data.tail(volume_rolling_win)
		print "\n stock return:", data['return']
		print "\n labels: ", y.shape, "\n", y#.tail(volume_rolling_win)


	# In[25]:

	## Split the data into training and testing sets using the given feature as the target
	## -----------------------------------------------
	
	from sklearn.cross_validation import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=42)

	if debug:
		print "y_test: \n", y_test

	# In[26]:

	### implement some useful functions
	### ================================================

	#get_ipython().magic(u'matplotlib inline')
	from time import time
	from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
	from sklearn.metrics import roc_curve, auc
	import matplotlib.pyplot as plt

	from sklearn.metrics import make_scorer
	from sklearn.grid_search import GridSearchCV

	def fa_score(ytrue, ypred, pos_label=1):
	    return min(f1_score(ytrue, ypred, pos_label=1), accuracy_score(ytrue, ypred))

	def train_classifier(clf, X_train, y_train, gridSearch=False, parameters=None):
	    ''' Fits a classifier to the training data. Can configure CV GridSearch'''
	    
	    # Start the clock, train the classifier, then stop the clock
	    start = time()
	    
	    print X_train.shape, y_train.shape
	    if not gridSearch:
	        clf.fit(X_train, y_train)
	    else:
	        f1_scorer = make_scorer(f1_score, pos_label=1)
        	accu_scorer = make_scorer(accuracy_score)
        	fa_scorer = make_scorer(fa_score)
        	grid_obj = GridSearchCV(clf, parameters, scoring = fa_scorer)
        	grid_obj.fit(X_train, y_train)
        	print "GridSearch Best Parameters: ", grid_obj.best_params_, '=', grid_obj.best_score_
        	clf = grid_obj.best_estimator_
	    
	    end = time()
	    
	    # Print the results
	    print "Trained model in {:.4f} seconds".format(end - start)
	    return clf
	    
	def predict_labels(clf, features, target):
	    ''' Makes predictions using a fit classifier. '''
	    
	    # Start the clock, make predictions, then stop the clock
	    start = time()
	    y_pred = clf.predict(features)
	    end = time()
	    
	    # Print and return results
	    print "Made predictions in {:.4f} seconds.".format(end - start)
	    #print "labels: \n", target.values
	    #print "preds: \n", y_pred
	    return f1_score(target.values, y_pred, pos_label=1), accuracy_score(target.values, y_pred),\
	            recall_score(target.values, y_pred), precision_score(target.values, y_pred)


	def train_predict(clf, X_train, y_train, X_test, y_test, gridSearch=False, parameters=None):
	    ''' Train and predict using a classifer based on F1 score. '''

	    # Indicate the classifier and the training set size
	    print "\n\nTraining a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
	    
	    # Train the classifier
	    clf = train_classifier(clf, X_train, y_train, gridSearch, parameters)
	    
	    # Print the results of prediction for both training and testing
	    f1train, atrain, rtrain, ptrain = predict_labels(clf, X_train, y_train)
	    print "For training set:",'\naccuracy =',atrain, '\nprecision =', ptrain, '\nrecall =', rtrain, '\nf1 =',f1train
	    f1test, atest, rtest, ptest = predict_labels(clf, X_test, y_test)
	    print "For testing set:",'\naccuracy =',atest, '\nprecision =', ptest, '\nrecall =', rtest, '\nf1 =',f1test
	    
	    return clf, min(f1test,atest)
	    
	def plotROC(clf, X_test, y_test):
	    # Determine the false positive and true positive rates
	    print "output labels belong to : ", clf.classes_
	    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
	    # Calculate the AUC
	    roc_auc = auc(fpr, tpr)
	    print 'ROC AUC: %0.2f' % roc_auc
	    # Plot of a ROC curve for a specific class
	    plt.figure()
	    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	    plt.plot([0, 1], [0, 1], 'k--')
	    plt.xlim([0.0, 1.0])
	    plt.ylim([0.0, 1.05])
	    plt.xlabel('False Positive Rate')
	    plt.ylabel('True Positive Rate')
	    plt.title('ROC Curve')
	    plt.legend(loc="lower right")
	    plt.show()
	    
	def compBestClf(clf_cand, score_cand, clf_best, score_best):
	    if score_cand > score_best: 
	        score_best = score_cand
	        clf_best = clf_cand
	    return clf_best, score_best

	### ================================================//


	# In[27]:

	### train and fit using a set of models and select the best
	### ================================================

	# Variables to compare and find best classifier model
	clf_best = None
	score_best = 0

	# In[28]:

	## Boost Classifier
	## -----------------------------------------------

	from sklearn.ensemble import AdaBoostClassifier

	# TODO: Initialize the three models
	clf_Bst = AdaBoostClassifier(n_estimators=20)

	# TODO: Execute the 'train_predict' function for each classifier and each training set size
	param_Bst = {'n_estimators': [2, 3, 5, 10, 20, 50], 'learning_rate': [0.1, 0.5, 1]}
	clf_Bst, f1_Bst = train_predict(clf_Bst, X_train, y_train, X_test, y_test, gridSearch=True, parameters = param_Bst)

	#plotROC(clf_Bst, X_test, y_test)

	# compare best clf
	clf_best, score_best = compBestClf(clf_Bst, f1_Bst, clf_best, score_best)


	# In[29]:

	## Random Forest Classifier
	## -----------------------------------------------

	from sklearn.ensemble import RandomForestClassifier

	clf_RF = RandomForestClassifier(n_estimators=10, n_jobs=-1)

	param_RF = {'n_estimators': [5, 10, 20, 40, 80, 160], 'max_depth': [2,3,4,8]}
	clf_RF, f1_RF = train_predict(clf_RF, X_train, y_train, X_test, y_test, gridSearch=True, parameters = param_RF)

	#plotROC(clf_RF, X_test, y_test)

	# compare best clf
	clf_best, score_best = compBestClf(clf_RF, f1_RF, clf_best, score_best)


	# In[30]:

	## Naive Bayes Classifier
	## -----------------------------------------------	

	from sklearn.naive_bayes import GaussianNB

	clf_NB = GaussianNB()

	clf_NB, f1_NB = train_predict(clf_NB, X_train, y_train, X_test, y_test)

	#plotROC(clf_NB, X_test, y_test)

	# compare best clf
	clf_best, score_best = compBestClf(clf_NB, f1_NB, clf_best, score_best)


	# In[31]:

	## Decision Tree Classifier
	## -----------------------------------------------

	from sklearn.tree import DecisionTreeClassifier

	clf_DT = DecisionTreeClassifier(random_state=0, min_samples_split=80)

	param_DT = {'min_samples_split': [2,5,10,50,100,200], 'max_depth': [2,3,5,8]}
	clf_DT, f1_DT = train_predict(clf_DT, X_train, y_train, X_test, y_test, gridSearch=True, parameters = param_DT)

	#plotROC(clf_DT, X_test, y_test)

	# compare best clf
	clf_best, score_best = compBestClf(clf_DT, f1_DT, clf_best, score_best)


	# In[32]:

	## SVM Classifier
	## -----------------------------------------------

	from sklearn import svm

	# TODO: Initialize the three models
	clf_SVM = svm.SVC(kernel='poly', probability=True)

	# TODO: Execute the 'train_predict' function for each classifier and each training set size
	param_SVM = {'C': [1,6,36], 'gamma': [0.5,2,4], 'degree': [2,3]}
	#clf_SVM, f1_SVM = train_predict(clf_SVM, X_train, y_train, X_test, y_test, gridSearch=True, parameters = param_SVM)	#[SLOW!!!]

	#plotROC(clf_SVM, X_test, y_test)

	# compare best clf
	#clf_best, score_best = compBestClf(clf_SVM, f1_SVM, clf_best, score_best)	# [SLOW!!!]


	# In[33]:

	## KNN Classifier
	## -----------------------------------------------

	from sklearn.neighbors import KNeighborsClassifier

	clf_KNN = KNeighborsClassifier(n_neighbors=4)

	param_KNN = {'n_neighbors':[2,4,8,16,32]}
	clf_KNN, f1_KNN = train_predict(clf_KNN, X_train, y_train, X_test, y_test, gridSearch=True, parameters = param_KNN)

	#plotROC(clf_KNN, X_test, y_test)
	             
	# compare best clf
	clf_best, score_best = compBestClf(clf_KNN, f1_KNN, clf_best, score_best)


	# In[34]:

	## Output Classifier Model Selection Sumary and Best Prediction Result
	## -----------------------------------------------

	print "\n\n===================MODEL TRAIN END======================="
	print "Best Classifier is: \n", clf_best
	print "Best Classifier's Fa Score on test: ", score_best

	if debug:
		plotROC(clf_best, X_test, y_test)

	x_endtime_row = x_endtime.reshape(1, -1)
	pred_v = clf_best.predict(x_endtime_row)
	pred_p = clf_best.predict_proba(x_endtime_row)
	pred_str = "RISE" if pred_v > 0 else "FALL"
	print "Predict: \n", company, "stock after", forcast, "days of", x_endtime.name,     "\nis going to (with confidence FALL/RISE =", pred_p, "):" , pred_str


	return pred_str, pred_p, score_best



