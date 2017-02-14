#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=10000)
t0 = time()

print "\nTraining SVM kernal rbf c=10000"
print "Training time:", round(time()-t0, 3), "s"
clf.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"

print "Testing time:", round(time()-t0, 3), "s"
anwser = clf.predict(features_test)
print "Testing time:", round(time()-t0, 3), "s"
numberOfOneAnwser = 0;

for num in anwser:
    if ( num == [1]):
        numberOfOneAnwser += 1

print "Number of test data that has value 1 is {0}".format(numberOfOneAnwser);
print "Element 10 = {0} Element 26 = {1} Element 50 = {2}.".format(anwser[10], anwser[26], anwser[50])

print "Estimate accuracy time:", round(time()-t0, 3), "s"
accuracy = clf.score(features_test,labels_test)
print "Estimate accuracy time:", round(time()-t0, 3), "s"

print "Accuracy is {0} \n".format(accuracy);

clg = SVC(kernel="linear")
t1 = time()

print "Training SVM kernal linear"
print "Training time:", round(time()-t1, 3), "s"
clg.fit(features_train, labels_train)
print "Training time:", round(time()-t1, 3), "s"

print "Estimate accuracy time:", round(time()-t1, 3), "s"
accuracy2 = clg.score(features_test,labels_test)
print "Estimate accuracy time:", round(time()-t1, 3), "s"

print "Accuracy is {0} \n".format(accuracy2);


#########################################################


