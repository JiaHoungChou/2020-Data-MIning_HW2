import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(123)

# Homework one
# 1. Write a computer program for Perceptron neural network without hidden layer and prove that it can solve the OR problem
# instead of XOR problem.

database= pd.DataFrame({"X_1": [0, 1, 0, 1], "X_2": [0, 1, 1, 1], "Label": [0, 1, 1, 1]})

Show_The_Figure= False
if Show_The_Figure== True:
    plt.figure(num= 1, figsize= (8, 6))
    plt.grid(True)
    plt.scatter(database.X_1[1: ], database.X_2[1: ], color= "blue")
    plt.scatter(database.X_1[0], database.X_2[0], color= "red")
    plt.show()

def Step_Function_Matrix(X_matrix):
    X_matrix= np.array(X_matrix)
    for i in range(X_matrix.shape[0]):
        for j in range(X_matrix.shape[1]):
            if X_matrix[i][j]> 0.0:
                X_matrix[i][j]= float(1)
            else:
                X_matrix[i][j]= float(0)
    return np.mat(X_matrix)

def Cost_Fuction(X, y):
    X= np.array(X)
    y= np.array(y)
    return np.mat(y- X)

def forward_(X, labels, eta, iteration):
    labels= np.mat(labels)
    X= np.hstack((np.ones((len(X), 1)), np.mat(X)))
    m, n= X.shape

    W_matrix= np.mat(np.random.random(size= (1, n)))
    B_matrix= np.mat(np.random.random(size=(m, 1)))
    
    for i in range(0, int(iteration)):
        Z= Step_Function_Matrix(X * W_matrix.T - B_matrix)
        loss = Cost_Fuction(Z, labels.T)
        
        W_matrix+= eta* loss.T* X
        B_matrix-= eta* loss

    return W_matrix, B_matrix


X_OR= np.array(database[["X_1", "X_2"]])
y_OR= np.array(database["Label"])

W_OR, B_OR= forward_(X= X_OR, labels= y_OR, eta= 0.1, iteration= 1)

OR_test_x_matrix= np.hstack((np.ones((len(X_OR), 1)), np.mat(X_OR)))
OR_test_label= Step_Function_Matrix(OR_test_x_matrix* W_OR.T- B_OR)

print("========== OR Problem ============")
print("*Test_label (Target= 0, 1, 1, 1):\n", np.array(OR_test_label).ravel(), "\n")
print("When training the model, the model can classify the OR problem.", "\n\n")


XOR_database= pd.DataFrame({"X_1": [0, 0, 1, 1], "X_2": [0, 1, 0, 1], "Label": [0, 1, 1, 0]})

X_XOR= np.array(XOR_database[["X_1", "X_2"]])
y_XOR= np.array(database["Label"])

W_XOR, B_XOR= forward_(X= X_XOR, labels= y_XOR, eta= 0.1, iteration= 1)

XOR_test_x_matrix= np.hstack((np.ones((len(X_XOR), 1)), np.mat(X_XOR)))
XOR_test_label= Step_Function_Matrix(XOR_test_x_matrix* W_XOR.T- B_XOR)

print("========== XOR Problem ===========")
print("*Test_label (Target= 0, 1, 1, 0):\n", np.array(XOR_test_label).ravel(), "\n")
print("Don't solve the XOR problem, when test the model.")