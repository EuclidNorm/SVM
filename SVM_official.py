from sklearn import *
from util import normailization_0_1,label_formatter
from sklearn.datasets import load_breast_cancer
import numpy as np


def SMO_train(gamma,C,x_train,y_train,ker_type):
    if ker_type=="rbf":
        classifier = svm.SVC(kernel="rbf", gamma=gamma, C=C, verbose=1,tol=1e-3)
    elif ker_type=="linear":
        classifier=svm.SVC(C=C,verbose=1,tol=1e-3)
    elif ker_type=="polynomial":
        #set c=1 by default
        c=1
        classifier=svm.SVC(kernel="poly",gamma=gamma,coef0=c,verbose=1,tol=1e-3)
    classifier.fit(x_train, y_train.ravel())
    return classifier
def SMO_test(classifier,x_test,y_test):
    predictions = classifier.predict(x_test)
    print("SMO-based SVM accuracy：", sum(1 - abs(predictions - y_test)) / len(y_test))
    return sum(1 - abs(predictions - y_test)) / len(y_test)

gamma=0.8
lamb=1



x,y=datasets.load_breast_cancer(return_X_y=True)
label_formatter(y)
x=normailization_0_1(x)





x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state =4, test_size = 0.2)
classifier=SMO_train(gamma,lamb,x_train,y_train,"rbf")

accuracy=SMO_test(classifier,x_test,y_test)



