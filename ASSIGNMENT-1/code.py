import numpy as np
import sklearn
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

# Function to compute the feature map for a single challenge
def feature_map(challenge):
    n = len(challenge)
    d = np.ones(n)
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= (1 - 2 * challenge[j])
        d[i] = product
    
    feature_vector = [1] + d.tolist()
    for k in range(len(d) - 1):
        feature_vector.append(d[k] * d[k + 1])
    
    return np.array(feature_vector)

def my_fit( X_train, y0_train, y1_train ):
    ################################
    # Non Editable Region Starting #
    ################################
    
    # Use this method to train your models using training CRPs
	# X_train has 32 columns containing the challenge bits
	# y0_train contains the values for Response0
	# y1_train contains the values for Response1
	
	# THE RETURNED MODELS SHOULD BE TWO VECTORS AND TWO BIAS TERMS
	# If you do not wish to use a bias term, set it to 0
    model = sklearn.svm.LinearSVC(C=100, tol=1e-5, penalty='l2', max_iter=1000, loss='squared_hinge', dual=False)
    feat = my_map(X_train)
    model.fit(feat, y0_train)
    w0 = model.coef_
    b0 = model.intercept_

    model.fit(feat, y1_train)
    w1 = model.coef_
    b1 = model.intercept_
    
    return w0, b0, w1, b1

################################
# Non Editable Region Starting #
################################
def my_map( X ):
    mapped_features = [feature_map(challenge) for challenge in X]
    return np.array(mapped_features)
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
