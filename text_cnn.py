#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import sys,os
import mxnet as mx
import numpy as np
import time
import math
import data_helpers
from collections import namedtuple

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # get a logger to accuracies are printed

logs = sys.stderr

CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

def make_text_cnn(sentence_size, num_embed, batch_size, vocab_size,
        num_label=2, filter_list=[2, 3, 4], num_filter=10,
        dropout=0., with_embedding=True):

    input_x = mx.sym.Variable('data') # placeholder for input
    input_y = mx.sym.Variable('softmax_label') # placeholder for output

    # embedding layer
    if not with_embedding:		#第一个看不懂的地方
        embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')
        conv_input = mx.sym.Reshape(data=embed_layer, target_shape=(batch_size, 1, sentence_size, num_embed))
    else:
        conv_input = input_x

    # create convolution + (max) pooling layer for each filter operation
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed), num_filter=num_filter)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1,1))
        pooled_outputs.append(pooli)

    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim=1)
    h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))

    # dropout layer
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool

    # fully connected
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')

    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

    # softmax output
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

    return sm


def setup_cnn_model(ctx, batch_size, sentence_size, num_embed, vocab_size,
        dropout=0.5, initializer=mx.initializer.Uniform(0.1), with_embedding=True):

    cnn = make_text_cnn(sentence_size, num_embed, batch_size=batch_size,
            vocab_size=vocab_size, dropout=dropout, with_embedding=with_embedding)
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


def train_cnn(model, X_train_batch, y_train_batch, X_dev_batch, y_dev_batch, batch_size,
        optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.0005, epoch=5):  #epoch=200
    m = model
    if not os.path.isdir('checkpoint'):
        os.system("mkdir checkpoint")
    # create optimizer
    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate

    updater = mx.optimizer.get_updater(opt)

    for iteration in range(epoch):
        tic = time.time()
        num_correct = 0
        num_total = 0
        for begin in range(0, X_train_batch.shape[0], batch_size):  #分批
            batchX = X_train_batch[begin:begin+batch_size]
            batchY = y_train_batch[begin:begin+batch_size]
            if batchX.shape[0] != batch_size:
                continue

            m.data[:] = batchX
            m.label[:] = batchY

            # forward
            m.cnn_exec.forward(is_train=True)

            # backward
            m.cnn_exec.backward()

            # eval on training data
            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

            # update weights
            norm = 0
            for idx, weight, grad, name in m.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm

            norm = math.sqrt(norm)
            for idx, weight, grad, name in m.param_blocks:
                if norm > max_grad_norm:
                    grad *= (max_grad_norm / norm)

                updater(idx, grad, weight)

                # reset gradient to zero
                grad[:] = 0.0

        # decay learning rate
        if iteration % 50 == 0 and iteration > 0:
            opt.lr *= 0.5
            print >> logs, 'reset learning rate to %g' % opt.lr

        # end of training loop
        toc = time.time()
        train_time = toc - tic
        train_acc = num_correct * 100 / float(num_total)

        # saving checkpoint
        if (iteration + 1) % 10 == 0:
            prefix = 'cnn'
            m.symbol.save('checkpoint/%s-symbol.json' % prefix)  #mkdir checkpoint文件夹
            save_dict = {('arg:%s' % k) :v  for k, v in m.cnn_exec.arg_dict.items()}
            save_dict.update({('aux:%s' % k) : v for k, v in m.cnn_exec.aux_dict.items()})
            param_name = 'checkpoint/%s-%04d.params' % (prefix, iteration)
            mx.nd.save(param_name, save_dict)
            print >> logs, 'Saved checkpoint to %s' % param_name


        # evaluate on dev set
        num_correct = 0
        num_total = 0
        for begin in range(0, X_dev_batch.shape[0], batch_size):
            batchX = X_dev_batch[begin:begin+batch_size]
            batchY = y_dev_batch[begin:begin+batch_size]

            if batchX.shape[0] != batch_size:
                continue

            m.data[:] = batchX
            m.cnn_exec.forward(is_train=False)

            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        dev_acc = num_correct * 100 / float(num_total)
        print >> logs, 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f \
                --- Dev Accuracy thus far: %.3f' % (iteration, train_time, train_acc, dev_acc)
    return m

def predict():
    pass

def train_without_pretrained_embedding():
    x, y, vocab, vocab_inv = data_helpers.load_data()
    vocab_size = len(vocab)


    x_train, x_dev, y_train, y_dev =data_helpers.train_test_split(x,y)


    print 'Train/Dev split: %d/%d' % (len(y_train), len(y_dev))
    print 'train shape:', x_train.shape
    print 'dev shape:', x_dev.shape
    print 'vocab_size', vocab_size
   
    batch_size = 100
    num_embed = 50
    sentence_size = x_train.shape[1]

    print 'batch size', batch_size
    print 'sentence max words', sentence_size
    print 'embedding size', num_embed

    cnn_model = setup_cnn_model(mx.cpu(), batch_size, sentence_size, num_embed, vocab_size, dropout=0.5, with_embedding=False) #原来是mx.gpu(0)
    m=train_cnn(cnn_model, x_train, y_train, x_dev, y_dev, batch_size)

    #m.data[:] = x_dev[:100]
    #m.cnn_exec.forward(is_train=False)#.predict('12345678')
    #print np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1)
    return m,batch_size,vocab

if __name__ == '__main__':
    m,batchsize,vocab=train_without_pretrained_embedding()
    
#    output = open('vocab.pkl', 'wb')
#    pickle.dump(vocab,output)
#    output.close()
    
#output = open('model.pkl', 'wb')
#pickle.dump(m.param_blocks,output)
#output.close()
    
#    sentence=data_helpers.load_test_data()
#    sentences_padded = data_helpers.pad_sentences(sentence)
#    sentence_test=[]
#    for sent in sentences_padded:
#        l=[]
#        for word in sent:
#            if word in vocab:
#                l.append(vocab[word])
#            else:
#                l.append(0)
#        sentence_test.append(l)
#    sentence_test=np.array(sentence_test)
#    print sentence_test[10]
#    l=[]
#    for begin in range(0,sentence_test.shape[0] ,100):
#        batch=sentence_test[begin:begin+100]
#        if batch.shape[0]!=100:
#            break
#        m.data[:]=batch
#        m.cnn_exec.forward(is_train=False)
#        l.extend(np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
#    f=open('1')
#    lines=f.readlines()
#    f.close()
#    f=open('2','w')
#    for i in xrange(len(l)):
#        if l[i]==0:
#            f.write(lines[i])
#    f.close()
            
#m.symbol.save('symbol')
