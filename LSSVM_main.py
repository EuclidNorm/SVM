import math
import numpy as np
from LSSVM_library import *
from sklearn import datasets
from sklearn import model_selection
from util import normailization_0_1,label_formatter
gamma=0.0002
lamb=2



x,y=datasets.load_breast_cancer(return_X_y=True)

for i in range(len(y)):
    if y[i]==0:
        y[i]=-1


#x=normailization_0_1(x)

print("Finished Preprocessing data.")



x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state =5, test_size = 0.2)
K=gen_kernel_mat_parallel(x_train,"rbf",gamma)
b_a=LSSVM_train(lamb,K,y_train)

acc=LSSVM_eval(b_a,gamma,x_test,y_test,x_train)

print(acc)