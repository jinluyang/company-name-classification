# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 17:01:30 2016

@author: root
"""
from collections import namedtuple
import pickle
import data_helpers
import mxnet as mx
import numpy as np

CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

def load_symbol():
    prefix = './checkpoint/cnn'
    model_loaded = mx.model.FeedForward.load(prefix, 2)
    model_loaded.numpy_batch_size=100
    return model_loaded

def load_param():
    pkl_file = open('model.pkl', 'rb')
    param_blocks  = pickle.load(pkl_file)
    pkl_file.close()
    return param_blocks



pkl_file = open('vocab.pkl', 'rb')
vocab  = pickle.load(pkl_file)
pkl_file.close()



sentence=data_helpers.load_test_data()
sentences_padded = data_helpers.pad_sentences(sentence)
sentence_test=[]
for sent in sentences_padded:
    l=[]
    for word in sent:
        if word in vocab:
            l.append(vocab[word])
        else:
            l.append(0)
    sentence_test.append(l)
sentence_test=np.array(sentence_test)

vocab_size = len(vocab)
num_embed = 50
batch_size = 100
sentence_size = sentence_test.shape[1]


#l=[]
#for begin in range(0,sentence_test.shape[0] ,100):
#    batch=sentence_test[begin:begin+100]
#    if batch.shape[0]!=100:
#        break
#    model_loaded.data[:]=batch
#    m.cnn_exec.forward(is_train=False)
#    l.extend(np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))