from torch.autograd import Variable
from utils import *
import random as random

from item_response import *
from neural_network import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from sklearn.impute import KNNImputer

import torch

from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
import matplotlib.pyplot as plt

import numpy as np
import csv
import os

def evaluateBAG(valid_data,zero_train_matrix,model_1nn,theta_1,beta_1, theta_2, beta_2):
  total = 0
  correct = 0
  
  model_1nn.eval()
  
  for i,u in enumerate(valid_data["user_id"]):
    inputs = Variable(train_data[u]).unsqueeze(0)
    output_1 = model_1(inputs)
    u = valid_data["user_id"][i]
    x_1 = (theta_1[u] - beta_1[q]).sum()   
    x_2 = (theta_2[u] - beta_2[q]).sum()

    guess = (output_1[0][valid_data["question_id"][i]].item()+sigmoid(x_1)+sigmoid(x_2))/3 >= 0.5
    if guess == valid_data["is_correct"][i]:
        correct += 1
        total += 1
    return correct / float(total)

def evaluateITR(data,theta_1,beta_1,theta_2,beta_2,theta_3,beta_3):
  pred = []
  for i, q in enumerate(data["question_id"]):
      u = data["user_id"][i]
      x_1 = (theta_1[u] - beta_1[q]).sum()
      x_2 = (theta_2[u] - beta_2[q]).sum()
      x_3 = (theta_3[u] - beta_3[q]).sum()
      p_a_1 = sigmoid(x_1)
      p_a_2 = sigmoid(x_2)
      p_a_3 = sigmoid(x_3)
      pred.append(((p_a_1+p_a_2+p_a_3)/3) >= 0.5)
  return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

zero_train_matrix, train_matrix, valid_data, test_data = load_data()

train_data = load_train_csv("../data")
val_data = load_valid_csv("../data")
test_dat = load_public_test_csv("../data")


k = 50
lr = 0.001
num_epoch = 430
lamb = 1

lr_irt = 0.01
num_iterations = 15

#values_1 = np.random.randint(0,int(train_matrix.shape[0]),int(train_matrix.shape[0]))

val_acc_lst = []

#train_matrix_1 = train_matrix[values_1,:]
#zero_train_matrix_1 = zero_train_matrix[values_1,:]


train_matrix_2 = train_data.copy()
random.seed(30)
train_matrix_2['user_id'] = random.choices(train_matrix_2['user_id'],k = len(train_matrix_2['user_id']))
random.seed(30)
train_matrix_2['question_id'] = random.choices(train_matrix_2['question_id'],k = len(train_matrix_2['question_id']))
random.seed(30)
train_matrix_2['is_correct'] = random.choices(train_matrix_2['is_correct'],k = len(train_matrix_2['is_correct']))

train_matrix_3 = train_data.copy()
random.seed(60)
train_matrix_3['user_id'] = random.choices(train_matrix_3['user_id'],k = len(train_matrix_3['user_id']))
random.seed(60)
train_matrix_3['question_id'] = random.choices(train_matrix_3['question_id'],k = len(train_matrix_3['question_id']))
random.seed(60)
train_matrix_3['is_correct'] = random.choices(train_matrix_3['is_correct'],k = len(train_matrix_3['is_correct']))

train_matrix_4 = train_data.copy()
random.seed(90)
train_matrix_4['user_id'] = random.choices(train_matrix_4['user_id'],k = len(train_matrix_4['user_id']))
random.seed(90)
train_matrix_4['question_id'] = random.choices(train_matrix_4['question_id'],k = len(train_matrix_4['question_id']))
random.seed(90)
train_matrix_4['is_correct'] = random.choices(train_matrix_4['is_correct'],k = len(train_matrix_4['is_correct']))



c_matrix_2 = np.zeros((542, 1774))
in_data_matrix_2 = np.zeros((542, 1774))

users_2 = train_matrix_2["user_id"]
questions_2 = train_matrix_2["question_id"]
is_correct_2 = train_matrix_2["is_correct"]


c_matrix_3 = np.zeros((542, 1774))
in_data_matrix_3 = np.zeros((542, 1774))


c_matrix_4 = np.zeros((542, 1774))
in_data_matrix_4 = np.zeros((542, 1774))

users_3 = train_matrix_3["user_id"]
questions_3 = train_matrix_3["question_id"]
is_correct_3 = train_matrix_3["is_correct"]

users_4 = train_matrix_4["user_id"]
questions_4 = train_matrix_4["question_id"]
is_correct_4 = train_matrix_4["is_correct"]
for i in range(len(users_2)):
  u_2 = users_2[i]
  q_2 = questions_2[i]
  c_2 = is_correct_2[i]
  u_3 = users_3[i]
  q_3 = questions_3[i]
  c_3 = is_correct_3[i]
  u_4 = users_4[i]
  q_4 = questions_4[i]
  c_4 = is_correct_4[i]
  
   
  c_matrix_2[u_2, q_2] = c_2
  in_data_matrix_2[u_2, q_2] = 1  
  c_matrix_3[u_3, q_3] = c_3
  in_data_matrix_3[u_3, q_3] = 1
  c_matrix_4[u_4, q_4] = c_4
  in_data_matrix_4[u_4, q_4] = 1

#model_1 = AutoEncoder(train_matrix_1.shape[1], k=k)

theta_1, beta_1, val_acc_lst = irt(train_matrix_2, val_data, lr_irt, num_iterations,c_matrix_2,in_data_matrix_2)
theta_2, beta_2, val_acc_lst = irt(train_matrix_3, val_data, lr_irt, num_iterations,c_matrix_3,in_data_matrix_3)
theta_3, beta_3, val_acc_lst = irt(train_matrix_4, val_data, lr_irt, num_iterations,c_matrix_4,in_data_matrix_4)
val_acc = evaluateITR(val_data,theta_1,beta_1,theta_2,beta_2,theta_3,beta_3)
print("val acc")
print(val_acc)
#train(model_1, lr, lamb, train_matrix_1, zero_train_matrix_1,
      #valid_data, num_epoch)



test_acc = evaluateITR(test_dat, theta_1,beta_1,theta_2,beta_2,theta_3,beta_3)
print("Test accuaracy =")
print(test_acc)