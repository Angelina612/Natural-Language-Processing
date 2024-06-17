#!/usr/bin/env python
# coding: utf-8

# In[153]:


import os
import utils
import NNs
import importlib
import torch

import pandas as pd
import seaborn as sns

importlib.reload(utils)
importlib.reload(NNs)

from matplotlib import pyplot as plt
from utils import TextDataset
from NNs import FFNN, RNN, MulticlassFFNN, MulticlassRNN
from torch.utils.data import DataLoader 
from sklearn.metrics import classification_report, confusion_matrix


# In[34]:


def plot_loss(loss):
    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over time')
    plt.show()


# ## IMDB Dataset

# In[59]:


def read_imdb_data(dir):
    data = []
    for i, label in enumerate(['pos', 'neg']):
        for file in os.listdir(os.path.join(dir, label)):
            with open(os.path.join(dir, label, file), 'r') as f:
                text = f.read()
                data.append({'text': text, 'label': i})
    return data


# In[60]:


train = read_imdb_data("aclImdb/train/")
test = read_imdb_data("aclImdb/test/")


# In[61]:


train = train[:1000] + train[-1000:]
test = test[:100] + test[-100:]


# In[62]:


train_data = TextDataset(train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = TextDataset(test, max_len=train_data.max_len)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


# In[63]:


print(train_data[0])


# In[64]:


torch.save(train_loader, 'imdb_train_loader.pt')
torch.save(test_loader, 'imdb_test_loader.pt')


# In[65]:


train_data.max_len


# ### Binary Class Feed Forward NN

# In[87]:


input_size = train_data.max_len 
output_size = 1

ffnn_model = FFNN(input_size, hidden_size_1=256, hidden_size_2=128, output_size=output_size, activation=torch.sigmoid)


# In[90]:


loss = utils.train(ffnn_model, train_loader, torch.nn.BCELoss(), epochs=10, lr=0.01)


# In[91]:


plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[96]:


pred, true = utils.test(ffnn_model, test_loader)


# In[97]:


print(pred, true)


# In[102]:


print(classification_report(true, pred))

cm = confusion_matrix(true, pred)
sns.heatmap(cm, xticklabels=["pos", "neg"], yticklabels=["pos", "neg"])

plt.ylabel('True')
plt.xlabel('Predicted')


# ### Binary Class RNN

# In[104]:


rnn_model = RNN(input_size, hidden_size=256, output_size=output_size, activation=torch.sigmoid)

loss = utils.train(rnn_model, train_loader, torch.nn.BCELoss(), epochs=10, lr=0.01)


# In[105]:


plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[106]:


pred, true = utils.test(rnn_model, test_loader)


# In[141]:


print(classification_report(true, pred))

cm = confusion_matrix(true, pred)
sns.heatmap(cm, xticklabels=["pos", "neg"], yticklabels=["pos", "neg"])

plt.ylabel('True')
plt.ylabel('Predicted')


# ## AG_News Dataset

# In[108]:


train = pd.read_csv("AG_News/train.csv")
train.head()


# In[109]:


def read_AG_news_data(file_path):
    data = []

    df = pd.read_csv(file_path)
    for i, row in df.iterrows():
        data.append({'text': row['Title'], 'label': row['Class Index'] - 1})

    return data


# In[110]:


train = read_AG_news_data("AG_News/train.csv")
test = read_AG_news_data("AG_News/test.csv")


# In[112]:


train = train[:1000] + train[-1000:]
test = test[:100] + test[-100:]


# In[113]:


train_data = TextDataset(train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

test_data = TextDataset(test, max_len=train_data.max_len)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)


# In[114]:


torch.save(train_loader, 'ag_news_train_loader.pt')
torch.save(test_loader, 'ag_news_test_loader.pt')


# ### Multi Class Feed Forward NN

# In[134]:


ffnn_model = MulticlassFFNN(input_size=train_data.max_len, hidden_size_1=256, hidden_size_2=128, output_size=4)

loss = utils.train(ffnn_model, train_loader, torch.nn.CrossEntropyLoss(), epochs=10, lr=0.01)


# In[135]:


plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[136]:


pred, true = utils.test(ffnn_model, test_loader)


# In[142]:


print(classification_report(true, pred))

cm = confusion_matrix(true, pred)
sns.heatmap(cm, xticklabels=[1,2,3,4], yticklabels=[1,2,3,4])

plt.ylabel('True')
plt.xlabel('Predicted')


# ### Multi Class RNN

# In[154]:


rnn_model = MulticlassRNN(input_size = train_data.max_len, hidden_size=256, output_size=4)

loss = utils.train(rnn_model, train_loader, torch.nn.CrossEntropyLoss(), epochs=10, lr=0.01)


# In[155]:


plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[156]:


pred, true = utils.test(rnn_model, test_loader)


# In[158]:


print(classification_report(true, pred))

cm = confusion_matrix(true, pred)
sns.heatmap(cm, xticklabels=["pos", "neg"], yticklabels=["pos", "neg"])

plt.ylabel('True')
plt.xlabel('Predicted')

