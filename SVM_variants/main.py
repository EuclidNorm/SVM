import numpy as np
from sklearn import model_selection
from sklearn import datasets
from PEGASOS_library import *

#this function serves as a normalization function on original data
def normailization_0_1(mtx):
    #assume that each line is an individual data
    output_array=np.zeros((len(mtx),len(mtx[0])))
    for i in range(len(mtx[0])):
        col_max=mtx[0][i]
        col_min=mtx[0][i]

        for j in range(len(mtx)):
            if mtx[j][i]<col_min:
                col_min=mtx[j][i]
            elif mtx[j][i]>col_max:
                col_max=mtx[j][i]
        if col_max==col_min:
            for j in range(len(mtx)):
                output_array[j][i] = (mtx[j][i] - col_min)
            continue
        for j in range(len(mtx)):
            output_array[j][i]=(mtx[j][i]-col_min)/(col_max-col_min)
    return output_array


#this function preprocesses the label to standard -1 and 1
def label_formatter(label):
    label_type=set()
    for i in range(len(label)):
        if label[i] not in label_type:
            label_type.add(label[i])
    if len(label_type)!=2:
        print("Warning: Multiple classes beyond 2. This code only supports SVM of 2 classes.")
    label_0=label_type.pop()
    label_1=label_type.pop()
    for i in range(len(label)):
        if label[i]==label_0:
            label[i]=1
        elif label[i]==label_1:
            label[i]=-1
    return label



#setting the minibatch size
minibatch=32
#iteration times of PEGASOS
iteration_PEGASOS=2000

#hyper-parameter for rbf kernel function computation
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





