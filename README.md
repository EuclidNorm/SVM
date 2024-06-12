# SVM
This repo includes SMO-based SVM and two SVM variants, PEGASOS and LSSVM. The major difference between them and standard SVM based on SMO lies in the design of loss function. The code is written in a manner that is easy to understand and friendly to newcomers.

# Running Guidelines
The main function of SMO-based SVM, PEGASOS and LSSVM is respectively SVM_official.py, PEGASOS_main.py and LSSVM_main.py. You can just create a new Python project and put all the files in this repo into your project folder. Directly running the three files will work.
## Environment
* Windows (Simply use Pycharm)
* Python 3.10 (It has been tested that using Python 3.7 will encounter error in joblib package)
* scikit-learn 1.3.2
* numpy, math, joblib...

# SVM Introduction

## Standard SVM
The common objective function (also called as loss function) for standard SVM is:
![image](https://github.com/ChiangyuMo/SVM/assets/70008102/f6d27bc0-f812-4f2e-bd70-b6809faab5ff)
Where w refers to parameters to be trained, b the bias term, C a hyperparameter that controls the extent to tolerate wrong classification and ξ the tolerated error. SMO is usually used to minimize the loss. By using Lagrange multiplier we can obtain the dual form of the problem:
![image](https://github.com/ChiangyuMo/SVM/assets/70008102/2588ef44-53b8-4870-8eee-8d48d017c39a)


Where c is the Lagrange operator to be solved. Note that K(i,j) is the kernel function of xi and xj, which is xi*xj when applying linear kernel. The new decision function is now:
![image](https://github.com/ChiangyuMo/SVM/assets/70008102/be7fe3e9-92b3-40ab-b841-df8785d6b319)

## PEGASOS SVM
PEGASOS tries to eliminate the constraint terms, and such attempt makes sense since non-constraint convex optimization is usually easier to solve. The objective function for PEGASOS is:
![image](https://github.com/ChiangyuMo/SVM/assets/70008102/0ffa036a-4987-4a5b-b819-096b648ad7a2)
The idea behind such design is that PEGASOS brings the constraint term of the common objective function shown above into the objective function by adding a hinge loss. It is proven that the new loss is still a convex function, so gradient descent with dynamic step size is used to optimize the objective funciton. The dual of the objective function is:
![image](https://github.com/ChiangyuMo/SVM/assets/70008102/c98c8d3b-54da-45a0-ae84-c818363e4b3c)


## LSSVM
LSSVM rather turns the minimization problem into an equation-solving problem. It tries to solve the following minimization problem:
<div align=center>
<img src="https://github.com/ChiangyuMo/SVM/assets/70008102/dc57385b-1557-4802-81c0-42536d498898" width=500>
</div>
<div align=center>
<img src="https://github.com/ChiangyuMo/SVM/assets/70008102/d5cc1e92-aca0-4729-a2c0-11697f0a0ea8" width=500>
</div>
Where γ is a hyperparameter similar to C mentioned above.There are no longer any inequality constraints, and by constructing its Lagrangian and KKT conditions we obtain:
<div align=center>
<img src="https://github.com/ChiangyuMo/SVM/assets/70008102/4cf58b07-4615-46ea-8ca3-e6acc51169e3" width=500>
</div>
Where Ω(i,j)=K(i,j)*yi*yj, K is the kernel function, α the Lagrange operator to be solved, b the bias term to be solved. The linear system could be easily solved.






