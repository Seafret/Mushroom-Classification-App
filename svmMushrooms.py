"""
CPSC 599.44 Final Project - SVM Model (Mushroom Classification)
Professor: Dr. Richard Zhao
Authors: Lucas Rasmos-Strankman, Jessica Rogi, Nanjia Wang
Date: April 13, 2021
Summary: This is the code used to create the SVM Model for the final project.
         First the mushrooms.csv dataset is loaded and then reduced to only
         the chosen features. Data is then split and encoded, and ran through
         a loop to choosen the best parameters for the SVM model. Once the
         best parameters are found through cross-validation on the training
         split and the model is created and fit on the training set and
         aforementioned best parameters. The model is then exported and then
         finally, we can check the results of the fitted model on the
         previously unseen test set.
Required: joblib - to save model and encoder for mushroomapp.py
          pandas - for handling data set, etc.
          numpy - for dealing with arrays
          sklearn - for models, encoder, metrics, etc.
          
"""

# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from joblib import dump, load

#read data from csv
df = pd.read_csv('mushrooms.csv')

# get headers
header = list(df.columns.values)

# separate features from labels
y = df.loc[:,'class'].values
all_features = df.drop(['class'], axis=1)

# reduce dataset to chosen features
X = all_features[["cap-color", "bruises", "gill-attachment", "stalk-root", "habitat"]]

# set up encoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X)

# encode X
encoded_X = enc.transform(X)

# encode Y (posionous = 1, edible = 0)
# create array filled with zeros, same shape as y
encoded_y = np.zeros((8124,), dtype=int)
# loop through y, in any element is 'p' change same ele to 1 in encoded array
for i in range(len(y)):
  if y[i] == 'p':
    encoded_y[i] = 1

# loop through both, check conents is the same
copy_error_flag = False
for i in range(len(y)):
  if (y[i] == 'p' and encoded_y[i] == 0) or (y[i] == 'e' and encoded_y[i] == 1):
    copy_error_flag = True

if copy_error_flag:
  print("There was an error encoding!")
else:
  print("Array y was encoded correctly!")
  
# create train test splits for X and y  
#train test split, get the test data
X_rest, X_test, y_rest, y_test = train_test_split(X, encoded_y, test_size = 0.2, stratify = encoded_y)

# second train test split to get additional data needed
X_toss, X_keep, y_toss, y_keep = train_test_split(X_rest, y_rest, test_size = 0.29, stratify = y_rest)

# combine, X_rest and X_keep, and y_rest and y_keep into X_test and y_test respectively
X_train = np.concatenate((X_rest, X_keep))
y_train = np.concatenate((y_rest, y_keep))

# must encode Xs after splitting since I couldn't get sparse matrices to concatenate
X_train = enc.transform(X_train)
X_test = enc.transform(X_test)

# set up a list of kernels, and a list for their scores
my_kernels = ['linear', 'rbf', 'poly', 'sigmoid']  # maybe add precomputed later
kernel_score = [0, 0, 0, 0]
best_params = [(0, 0), (0, 0), (0, 0), (0, 0)]

# iterate through 
for i in range(len(my_kernels)):
  kernels_best_score = 0 # initialize best score
  kernels_best_params = (0.1, 0.1) # (C, gamma)
  my_C = 0.1 # initalize C
  while my_C <= 1000:
    my_gamma = 0.1 # initialize gamma
    while my_gamma <= 10:
      # train model on parameters from loops
      current_svm = SVC(C=my_C, gamma=my_gamma, kernel=my_kernels[i])
      current_score = cross_val_score(current_svm, X_train, y_train, cv=10, scoring='f1').mean()

      # update best score as required
      if current_score > kernels_best_score:
        kernels_best_score = current_score
        kernels_best_params = (my_C, my_gamma)

      # increment gamma
      my_gamma = my_gamma * 10

    # incrememnt C
    my_C = my_C * 100
  
  # update current bests before moving onto next kernel
  kernel_score[i] = kernels_best_score
  best_params[i] = kernels_best_params 

print(kernel_score)
print(best_params)

best_index = 0
best_score = 0
i = 0

# check which index has the best parameterse
for score in kernel_score:
  if score > best_score:
    best_score = score
    best_index = i
  i = i + 1

best_kernel = my_kernels[best_index]
best_C = best_params[best_index][0]
best_gamma = best_params[best_index][1]

print("Best kernel: " + best_kernel)
print("Best C: " + str(best_C))
print("Best gamma: " + str(best_gamma))

svm = SVC(kernel=best_kernel, C=best_C, gamma=best_gamma)
svm.fit(X_train, y_train)

# export model and encoder
dump(svm, 'mushmodel.joblib')
dump(enc, 'encoder.joblib')

# ONLY UNCOMMENT WHEN EXPORTING FINAL MODEL
print("SVM Model: Accuracy on test set: {:.2f}".format(svm.score(X_test, y_test)))

# show confusion matric
plot_confusion_matrix(svm, X_test, y_test)
plt.show()

# have model make prediction
test = np.array(['y', 't', 'f', 'c', 'g']) # edible sample
enc_test = enc.transform(test.reshape(1, -1))
is_poisonous = svm.predict(enc_test)[0]
if is_poisonous:
  print("Model Prediction: Poisonous")
else:
  print("Model Prediction: Edible")