from joblib import Parallel,delayed
import numpy as np
import math
import random



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





def minibatch_PEGASOS_kernelized(alpha,lamb,label,K,max_iteration,minibatch):
    iteration_True=max_iteration
    for iteration in range(1,max_iteration):
        it_set=random.sample(range(0,len(alpha)),minibatch)
        check=[]
        for i in range(minibatch):
            it_=it_set[i]
            alpha_temp=alpha*label
            check_term=np.dot(alpha_temp,K[it_])*label[it_]/(lamb*iteration)

            if check_term<1:
                check.append(it_)

        for i in range(len(check)):
            alpha[check[i]]+=1/(minibatch)

    return alpha

def minibatch_kernelized_PEGASOS_train(lamb,x_train,y_train,max_iteration,K,minibatch=32):

    alpha=np.zeros((len(x_train)))
    alpha_result=minibatch_PEGASOS_kernelized(alpha,lamb,y_train,K,max_iteration,minibatch)

    return alpha_result


def PEGASOS_test(alpha,x_train,x_test,y_train,y_test,lamb,ker_type,gamma):
    K_test2train=gen_kernel_mat_parallel_test(x_test,x_train,ker_type,gamma)
    K_test2train=np.transpose(K_test2train)
    alpha_temp=alpha*y_train

    score=np.dot(alpha_temp,K_test2train)
    label_pred=np.zeros((len(score)))
    for i in range(len(score)):
        if score[i]>0:
            label_pred[i]=1
        else:
            label_pred[i]=-1
    result=label_pred*y_test
    correct_predicton=np.where(result>0)
    print("The accuracy of PEGASOS is : ",len(correct_predicton[0])/len(y_test))





