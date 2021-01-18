from utils import *
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def neg_log_likelihood3(data, theta, beta, r, k):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param r: float
    :param k: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.

    for i in range(len(data["user_id"])):
      u = data["user_id"][i]
      q = data["question_id"][i]
      c = data["is_correct"][i]

      log_lklihood += np.log(1-r) - np.log(1 + np.exp(k[q] * (theta[u]-beta[q]))) - c*np.log(1-r) + c*np.log(r + np.exp(k[q] * (theta[u]-beta[q])))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood

def update3(data, lr, theta, beta, r, k, c_matrix, in_data_matrix):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param r: float
    :param k: Vector
    :param c_matrix: Matrix
    :param in_data_matrix: Matrix
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    t = theta.copy()
    b = beta.copy()
    r2 = r
    k2 = k.copy()
    diff = (np.subtract.outer(t[:,0], b[:,0]).T * k2).T
    tb_diff = np.subtract.outer(t[:,0], b[:,0]) * np.exp(diff) 
    r_diff = r2 + np.exp(diff)
    k_diff = (np.exp(diff).T * k2).T

    d_theta = c_matrix*(k_diff/r_diff) - k_diff/(1 + np.exp(diff))
    d_beta = np.dot((-1*d_theta*in_data_matrix).T, np.ones((542,1)))
    d_theta = np.dot(d_theta*in_data_matrix, np.ones((1774,1)))
    d_k = np.dot(((-1*tb_diff / (1+np.exp(diff)) + c_matrix*tb_diff/r_diff) * in_data_matrix).T, np.ones((542,1)))

    theta += lr * d_theta
    beta += lr * d_beta
    k = k + lr * d_k
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, r, k

def evaluate3(data, theta, beta, r, k, toprint):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param r: float
    :param k: Vector
    :toprint: Used for logging
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = r + (1-r)*sigmoid(k[q]*x)
        pred.append(p_a >= 0.5)

    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def irt3(data, val_data, lr, iterations, c_matrix, in_data_matrix):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.full((542, 1), 0.5)
    beta = np.full((1774, 1), 0.5)
    r = 0.3
    k = np.full((1774, 1), 1)
    
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood3(data, theta=theta[:,0], beta=beta[:,0], r=r, k=k[:,0])
        score = evaluate3(data=val_data, theta=theta[:,0], beta=beta[:,0], r=r, k=k[:,0], toprint=False)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, r, k = update3(data, lr, theta, beta, r, k, c_matrix, in_data_matrix)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, r, k, val_acc_lst

train_data = load_train_csv("../data")
# You may optionally use the sparse matrix.
sparse_matrix = load_train_sparse("../data")
val_data = load_valid_csv("../data")
test_data = load_public_test_csv("../data")

c_matrix = np.zeros((542, 1774))
in_data_matrix = np.zeros((542, 1774))

'''
train_data["user_id"].extend(val_data["user_id"])
train_data["user_id"].extend(test_data["user_id"])
train_data["question_id"].extend(val_data["question_id"])
train_data["question_id"].extend(test_data["question_id"])
train_data["is_correct"].extend(val_data["is_correct"])
train_data["is_correct"].extend(test_data["is_correct"])
'''
#####################################################################
# TODO:                                                             #
# Tune learning rate and number of iterations. With the implemented #
# code, report the validation and test accuracy.                    #
#####################################################################
users = train_data["user_id"]
questions = train_data["question_id"]
is_correct = train_data["is_correct"]
for i in range(len(users)):
  u = users[i]
  q = questions[i]
  c = is_correct[i]
  
  c_matrix[u, q] = c
  in_data_matrix[u, q] = 1
  
num_iterations = 26
lr = 0.01
theta, beta, r, k, val_acc_lst = irt3(train_data, val_data, lr, num_iterations, c_matrix, in_data_matrix)

fig1 = plt.figure()
ax = fig1.add_axes([0, 0, 1, 1])
ax.set_xlabel("Iterations")
ax.set_ylabel("Accuracy")
plt.plot([i+1 for i in range(num_iterations)], val_acc_lst, "-g", label="validation")
plt.legend(loc="upper right")
plt.show()


max_i = np.argmax(np.array(val_acc_lst))
print("The iteration value with the highest validation accuracy is " + str(max_i+1) + " with an accuracy of " + str(val_acc_lst[max_i]))
print("The train accuracy is " + str(evaluate3(train_data, theta[:,0], beta[:,0], r, k[:,0], True)))
print("The val accuracy is " + str(evaluate3(val_data, theta[:,0], beta[:,0], r, k[:,0], True)))
print("The test accuracy is " + str(evaluate3(test_data, theta[:,0], beta[:,0], r, k[:,0], True)))

#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################

#####################################################################
# TODO:                                                             #
# Implement part (c)                                                #
#####################################################################
fig1 = plt.figure()
ax = fig1.add_axes([0, 0, 1, 1])
ax.set_xlabel("Theta")
ax.set_ylabel("Probability")

j1 = []
j2 = []
j3 = []
j4 = []
j5 = []
# Plot questions 0, 100, 200, 300, 400
for i in range(542):
  j1.append(sigmoid(k[0]*(theta[i] - beta[0]).sum()))
  j2.append(sigmoid(k[100]*(theta[i] - beta[100]).sum()))
  j3.append(sigmoid(k[200]*(theta[i] - beta[200]).sum()))
  j4.append(sigmoid(k[300]*(theta[i] - beta[300]).sum()))
  j5.append(sigmoid(k[400]*(theta[i] - beta[400]).sum()))

plt.plot(theta, j1, ".")
plt.plot(theta, j2, ".")
plt.plot(theta, j3, ".")
plt.plot(theta, j4, ".")
plt.plot(theta, j5, ".")
plt.legend(["Q0","Q100","Q200","Q300","Q400"])
plt.show()
#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################
