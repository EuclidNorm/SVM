import numpy as np
from sklearn import model_selection
from sklearn import datasets
from PEGASOS_library import *
from util import *
#this function serves as a normalization function on original data



#setting the minibatch size
minibatch=64
#iteration times of PEGASOS
iteration_PEGASOS=200

#hyper-parameter for:
#the gamma of RBF kernel
#the exponential value of polynomial kernel
#not related to linear kernel
gamma=0.8



x,y=datasets.load_breast_cancer(return_X_y=True)
label_formatter(y)
x=normailization_0_1(x)


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,random_state=0, test_size=0.2)
K=gen_kernel_mat_parallel(x_train,"rbf",gamma)

data_num=len(x_train)
lamb_PEGASOS=1/data_num

alpha = minibatch_kernelized_PEGASOS_train(lamb_PEGASOS,x_train, y_train, iteration_PEGASOS, K,minibatch)
print("PEGASOS training phase complete")

PEGASOS_test(alpha,x_train,x_test,y_train,y_test,lamb_PEGASOS,"rbf",gamma)





