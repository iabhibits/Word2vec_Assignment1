#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sc
import pandas as pd


# In[14]:


f = open("SimLex-999.txt",'r')
lines = f.readline()
n= len(lines.split())
score= []
for i in range(n):
    a = [x for x in f.readline().split()]
    score.append(a)


# In[17]:


simplex_score = []
for i in range(n):
    x = score[i][0]
    y = score[i][1]
    z = score[i][3]
    d = {'word1':x,'word2':y,'cosine_score':z}
    print(d)
    simlex_score.append(d)


# In[21]:


print(simlex_score[0])


# In[25]:


import json
#data = {'word_to_index': word_to_index, 'index_to_word':index_to_word}
f=open('mapping.json','r')
data=json.loads(f.read())
word_to_index=data['word_to_index']
index_to_word=data['index_to_word']


# In[32]:


import tensorflow as tf
g = tf.Graph()
sess = tf.Session(graph=g)
saver = tf.train.import_meta_graph('word2vec.ckpt.meta', graph=g)
saver.restore(sess, 'word2vec.ckpt')
graph = sess.graph
embedding_matrix = graph.get_tensor_by_name('context_embeddings_2:0')
embedding_matrix = np.array(sess.run([embedding_matrix])[0])
embedding_matrix = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1).reshape(-1, 1)


# In[28]:


def sim(word1,word2,idx_to_wrd,wrd_to_idx,k=10):
    idx1=wrd_to_idx[word1]
    idx2=wrd_to_idx[word2]
    idcs=np.argsort(np.abs(np.dot(embedding_matrix[idx1],embedding_matrix[idx2])))[-k:]
    return [idx_to_wrd[i] for i in idcs][::-1]

