# SVM
This repo includes SMO-based SVM and two variants of standard SVM, PEGASOS and LSSVM. The major difference between them and standard SVM based on SMO is the design of loss function.

The most common objective function (also called as loss function) for SVM is:
![image](https://github.com/ChiangyuMo/SVM/assets/70008102/f6d27bc0-f812-4f2e-bd70-b6809faab5ff)
Where w refers to parameters to be trained, C a hyperparameter to be decided, b the bias term and ξ the tolerated error.It is also the objective function SMO tries to solve.

As for PEGASOS and LSSVM, both of them tries to eliminate the constraint terms. It makes sense since there are numerous optimization methods that work on non-constraint convex objective functions, including gradient descent. The objective function for PEGASOS is:
![image](https://github.com/ChiangyuMo/SVM/assets/70008102/0ffa036a-4987-4a5b-b819-096b648ad7a2)
The idea behind such design is that PEGASOS bring the constraint term of the common objective function above up into the objective function by adding a hinge loss. 
