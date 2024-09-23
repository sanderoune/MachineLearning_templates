# Multilayer perceptron classifier for Titanic survivors dataset.
# ML library: pyTorch.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("./train.csv")

data.info()
data.sample(3)

X, y = data.drop(['Survived'], axis = 1), data['Survived']

num_cols = [x for x in X.columns if data[x].dtype in ['int64', 'float64']]
cat_cols = [x for x in X.columns if data[x].dtype == 'object']

from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_transform = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='constant')),
                ('scale', MaxAbsScaler())
])

cat_transform = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scale', MaxAbsScaler())
])

preprocess = ColumnTransformer(transformers=[
                               ('cat', cat_transform, cat_cols),
                               ('num', num_transform, num_cols)
])

X = preprocess.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

X_train.shape, y_train.shape

X_train = X_train.toarray()
X_test = X_test.toarray()
y_train = y_train.values
y_test = y_test.values

print(y_test)
print(y_test.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.autograd import Variable

class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearRegression,self).__init__()
        self.f1 = nn.Linear(input_size, 2000)
        self.f2 = nn.Linear(2000, output_size)


    def forward(self,x):
        x = self.f1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p = 0.3)
        x = self.f2(x)
        return  F.sigmoid(x)

batch_size = 30
batch_no = len(X_train) // batch_size

X_train.shape

def generate_batches(X, y, batch_size):
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))

    for i in range(len(X)//batch_size):
        if i + batch_size >= len(X):
            continue
        ind = perm[i*batch_size : (i+1)*batch_size]
        yield (X[ind], y[ind])


input_dim = 1730
# The output dimension is 2 because output[0] = Prob(not survived) and output[1] = Prob(survived)
# So the output gives two numeber and we take the index of the maximum of the two.
output_dim = 2
learning_rate = 1
model = LinearRegression(input_dim,output_dim)
error = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.5)

loss_list = []
acc_list = []
iteration_number = 200

for iteration in range(iteration_number):
    batch_loss = 0
    batch_accur = 0
    temp = 0

    for (x, y) in generate_batches(X_train, y_train, batch_size):
        inputs = Variable(torch.from_numpy(x)).float()
        labels = Variable(torch.from_numpy(y))
            
        optimizer.zero_grad() 
        results = model(inputs)
        loss = error(results, labels)
        batch_loss += loss.data
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, pred = torch.max(results, 1)
            batch_accur += torch.sum(pred == labels)
            temp += len(pred)
    
    loss_list.append(batch_loss/batch_no)
    acc_list.append(batch_accur/temp)
    
    if(iteration % 50 == 0):
        print('epoch {}: loss {}, accuracy {}'.format(iteration, batch_loss/batch_no, batch_accur/temp))

plt.plot(range(iteration_number),loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.savefig('./training_loss_pytorch', dpi = 300)
plt.close()
plt.plot(range(iteration_number),acc_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.savefig('./training_accuracy_pytorch', dpi = 300)
plt.close()


X_test_var = Variable(torch.FloatTensor(X_test), requires_grad=True) 
with torch.no_grad():
    test_result = model(X_test_var)
values, labels = torch.max(test_result, 1)
survived = labels.data.numpy()
print((survived == y_test).sum()/len(survived))

X_test_origin = pd.read_csv("./test.csv")
submission = pd.read_csv("./gender_submission.csv")
X_test_origin = preprocess.transform(X_test_origin)
X_test_origin = X_test_origin.toarray()
X_test_var = Variable(torch.FloatTensor(X_test_origin), requires_grad=True) 
with torch.no_grad():
    test_result = model(X_test_var)
values, labels = torch.max(test_result, 1)
survived = labels.data.numpy()
X_test_1 = pd.read_csv("./test.csv")
'''
import csv

submission = [['PassengerId', 'Survived']]
for i in range(len(survived)):
    submission.append([X_test_1.PassengerId.loc[i], survived[i]])

with open('submission.csv', 'w') as submissionFile:
    writer = csv.writer(submissionFile)
    writer.writerows(submission)
    
print('Writing Complete!')
'''

