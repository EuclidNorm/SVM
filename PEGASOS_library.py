
import numpy as np
import math
import random
from util import gen_kernel_mat_parallel_test






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





