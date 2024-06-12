import numpy as np
from joblib import Parallel,delayed
import math
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



def para_compute(i,data,ker_type,gamma):
    num_data=len(data)
    K_line=np.zeros((num_data))
    if ker_type=="rbf":
        for j in range(i,num_data):
            vec=data[i]-data[j]
            vec=vec*vec
            summation=sum(vec)
            K_line[j]=math.exp(summation*-1*gamma)
    elif ker_type=="linear":
        for j in range(i,num_data):
            K_line[j]=np.dot(data[i],data[j])
    elif ker_type=="polynomial":
        c=1
        for j in range(i,num_data):
            #set c=1 by default
            xyplusc=np.dot(data[i],data[j])+c
            K_line[j]=pow(xyplusc,gamma)
    return K_line


def para_compute_test(i,data_1,data_2,ker_type,gamma):
    num_data=len(data_2)
    K_line=np.zeros((num_data))
    if ker_type=="rbf":
        for j in range(0,num_data):
            vec=data_1[i]-data_2[j]
            vec=vec*vec
            summation=sum(vec)
            K_line[j]=math.exp(summation*-1*gamma)
    elif ker_type=="linear":
        for j in range(0,num_data):
            K_line[j]=np.dot(data_1[i],data_2[j])
    elif ker_type=="polynomial":
        c=1
        for j in range(0,num_data):
            #set c=1 by default
            xyplusc=np.dot(data_1[i],data_2[j])+c
            K_line[j]=pow(xyplusc,gamma)
    return K_line

def gen_kernel_mat_parallel(data,ker_type,gamma=0.001):

    K_uptriangle=np.array(Parallel(n_jobs=-1)(delayed(para_compute)(i,data,ker_type,gamma) for i in range(len(data))))
    for i in range(len(K_uptriangle)):
        for j in range(0,i):
            K_uptriangle[i][j]=K_uptriangle[j][i]
    return K_uptriangle

def gen_kernel_mat_parallel_test(x_test,x_train,ker_type,gamma):
    K_uptriangle=np.array(Parallel(n_jobs=-1)(delayed(para_compute_test)(i,x_test,x_train,ker_type,gamma) for i in range(len(x_test))))
    return K_uptriangle