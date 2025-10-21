#coding=utf8
"""
Created on Thu Oct 12 17:48:23 2025

@author: Neal LONG

Hint max() is a built-in function in Python 
"""

import pickle

# %%  Define necessary functions


def linear_func(W,X):
    """
    Define a 2-dimensional discriminative linear function, using W[0] as the intercept.
    """
    return W[0]+W[1]*X[0]+W[2]*X[1]


def hinge_loss(f_x, y_true):
    """
    Calculate the hinge loss using the output of the linear 
     discriminative function and its corresponding true label y. 
    """

    # pass #++insert your code here++ to replace the "pass" statement here 
    # and complete the function definition 
    return max(0, 1 - f_x * y_true)

def zero_one_loss(f_x, y_true):
    """
    Calculate the zero-one loss using the output of the linear 
     discriminative function and its corresponding true label y. 
    """
    if f_x*y_true>=0:
        return 0
    else:
        return 1
    


# %%  Demo of computing zero-one loss on the data point (X, label) with linear function W
W = (-0.45236953, 2.23604794, -3.94803128)
X = [2.5, 3.7]
label = 1

f_x = linear_func(W, X)
loss = zero_one_loss(f_x, label)
print("zero_one_loss_total of linear function W on data point (X, label) is", loss)



with open('./data/Q1_features.pkl','rb') as rf:
    X = pickle.load(rf)

with open('./data/Q1_labels.pkl','rb') as rf:
    Y_true = pickle.load(rf)
print("Number of records = ", len(X), len(Y_true))

W1 = (-0.76862686, 1.50126098, -2.3948365 )
W2 = (-0.42862686, 1.50126098, -2.3948365 )
W3 = (-0.59862686, 1.50126098, -2.3948365 )

# %% #++insert your code here++ to
# 1. Based on the training dataset (X, Y_true) with 99 records, calculate the 
#    total zero-one loss and hinge loss for three distinct linear 
#    discrimination functions using weights W1, W2, and W3, respectively.
# 2. Compare the results and provide answers to the questions.
# Note:  Utilize the functions linear_func(), hinge_loss(), and zero_one_loss()
#     as defined above.
W = [W1, W2, W3]
loss_func = [hinge_loss, zero_one_loss]
for w in W:
    print("For W=", w)
    for lf in loss_func:
        total_loss = 0
        for i in range(len(X)):
            f_x = linear_func(w, X[i])
            total_loss += lf(f_x, Y_true[i])
        print("  Total", lf.__name__, "=", total_loss)

'''
Results:
For W= (-0.76862686, 1.50126098, -2.3948365)
  Total hinge_loss = 7.647114319999996
  Total zero_one_loss = 0
For W= (-0.42862686, 1.50126098, -2.3948365)
  Total hinge_loss = 9.109041068000018
  Total zero_one_loss = 0
For W= (-0.59862686, 1.50126098, -2.3948365)
  Total hinge_loss = 7.762552864000006
  Total zero_one_loss = 0
'''

'''
The results show that all three linear functions achieve a total zero-one loss of 0, 
indicating perfect classification on the training dataset. 
However, the hinge loss varies among the three functions, with W1 having the lowest hinge loss, 
suggesting it has the largest margin of separation between classes. W2 has the highest hinge loss, 
indicating a smaller margin. This demonstrates that while zero-one loss indicates classification accuracy, 
hinge loss provides insight into the confidence of those classifications.
'''