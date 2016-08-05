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


def setup_cnn_model(ctx, batch_size, sentence_size, num_embed, vocab_size,
        dropout=0.5, initializer=mx.initializer.Uniform(0.1), with_embedding=True):

    cnn=mx.sym.load('./symbol')
    arg_names = cnn.list_arguments()

    input_shapes = {}
    if with_embedding:
        input_shapes['data'] = (batch_size, 1, sentence_size, num_embed)
    else:
        input_shapes['data'] = (batch_size, sentence_size)

    arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)	#第二个看不懂
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        if name in ['softmax_label', 'data']: # input, output
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)

    cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

    param_blocks = []
    arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
    for i, name in enumerate(arg_names):
        if name in ['softmax_label', 'data']: # input, output
            continue
        initializer(name, arg_dict[name])

        param_blocks.append( (i, arg_dict[name], args_grad[name], name) )

    out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))

    data = cnn_exec.arg_dict['data']
    label = cnn_exec.arg_dict['softmax_label']

    return CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, label=label, param_blocks=param_blocks)

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

cnn_model = setup_cnn_model(mx.cpu(), batch_size, sentence_size, num_embed, vocab_size, dropout=0.5, with_embedding=False) 

l=[]
for begin in range(0,sentence_test.shape[0] ,100):
    batch=sentence_test[begin:begin+100]
    if batch.shape[0]!=100:
        break
    cnn_model.data[:]=batch
    cnn_model.cnn_exec.forward(is_train=False)
    l.extend(np.argmax(cnn_model.cnn_exec.outputs[0].asnumpy(), axis=1))