import numpy as np
import math
from joblib import Parallel,delayed

def para_compute(i,data,ker_type,gamma):
    num_data=len(data)
    K_line=np.zeros((num_data))
    if ker_type=="rbf":
        for j in range(i,num_data):
            vec=data[i]-data[j]
            vec=vec*vec
            summation=sum(vec)
            K_line[j]=math.exp(summation*-1*gamma)
    return K_line




def gen_kernel_mat_parallel(data,ker_type,gamma=0.001):
    if ker_type=="rbf":
        K_uptriangle=np.array(Parallel(n_jobs=-1)(delayed(para_compute)(i,data,ker_type,gamma) for i in range(len(data))))
        for i in range(len(K_uptriangle)):
            for j in range(0,i):
                K_uptriangle[i][j]=K_uptriangle[j][i]
    return K_uptriangle

#Below is not a support vector machine that enforces points to be out of the two hyperplane.
#It is a classification machine that tries to merely find the hyperplane that has the best training accuracy.
#What is strange is that this machine completely outperforms SVM in WBC dataset


# def gen_Equation(K,lamb,y):
#     sub_A = K + np.eye(len(K)) / lamb
#     A = np.ones((len(K) + 1, len(K) + 1))
#     B = np.zeros((len(K) + 1))
#     A[0][0] = 0
#     for i in range(1, len(A)):
#         for j in range(1, len(A)):
#             A[i][j] = sub_A[i - 1][j - 1]
#     for i in range(1, len(B)):
#         B[i] = y[i - 1]
#     return A,B

def gen_Equation(K,lamb,y):
    yy=np.resize(y,(len(y),1))
    yyy=np.dot(yy,yy.transpose())
    new_K=np.multiply(yyy,K)
    sub_A = new_K + np.eye(len(K))/lamb
    A = np.zeros((len(K) + 1, len(K) + 1))
    B = np.ones((len(K) + 1))
    B[0]=0
    for i in range(1, len(A)):
        for j in range(1, len(A)):
            A[i][j] = sub_A[i - 1][j - 1]
    for i in range(1,len(A)):
        A[0][i]=-y[i-1]
        A[i][0]=y[i-1]
    return A,B

def predict_rbf(b_a,x_test,x_train,gamma):
    score=0
    score+=b_a[0]
    for i in range(len(x_train)):
        v=x_test-x_train[i]
        k_term=np.exp(-1*gamma*sum(v*v))
        score+=k_term*b_a[i+1]
    if score>=0:
        return 1
    elif score<0:
        return -1



def LSSVM_train(lamb,K,y_train):
    A, B = gen_Equation(K, lamb, y_train)
    b_a = np.linalg.solve(A, B)
    return b_a


def LSSVM_eval(b_a,gamma,x_test,y_test,x_train):
    count = 0
    for i in range(len(x_test)):
        if (predict_rbf(b_a, x_test[i], x_train, gamma) == y_test[i]):
            count += 1
    return count / len(x_test)