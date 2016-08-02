# -*- coding: utf-8 -*-
import numpy as np
import re
import itertools
from collections import Counter
import jieba
# from gensim.models import word2vec

def remove_bracket(x):
    eng_brk = re.compile('\(.*?\)')
    chi_brk = re.compile('\（.*?\）')
    brk = re.compile('\【.*?\】')
    x = eng_brk.sub('', x)
    x = chi_brk.sub('', x)
    x = brk.sub('', x)
    return x

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    string = re.sub(r"\'s", " \'s", string)
    
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = remove_bracket(string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("normal.csv").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("rubbish.csv").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [jieba.lcut(s) for s in x_text]
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_multilabels():
    """
    5 kinds of company name
    """
    # Load data from files
    positive_examples = list(open("normal.csv").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("rubbish.csv").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    invest_examples = list(open("invest.csv").readlines())
    invest_examples = [s.strip() for s in invest_examples]
    sci_examples = list(open("science.csv").readlines())
    sci_examples = [s.strip() for s in sci_examples]
    serv_examples = list(open("service.csv").readlines())
    serv_examples = [s.strip() for s in serv_examples]
    # Split by words
    x_text = positive_examples + negative_examples+invest_examples+sci_examples+serv_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [jieba.lcut(s) for s in x_text]
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    invest_labels = [2 for _ in invest_examples]
    sci_labels = [3 for _ in sci_examples]
    serv_labels = [4 for _ in serv_examples]
    y = np.concatenate([positive_labels, negative_labels, invest_labels, sci_labels, serv_labels], 0)
    return [x_text, y]

def pad_sentences(sentences, padding_word="</s>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = 50#max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()	#改为多分类
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def train_test_split(x,y,rate=0.8):
    labels = Counter(y)
    d=dict(labels)
    num_labels=len(labels)
    # randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
     # split train/dev set
    x_dev = []
    y_dev = []
    x_train = []
    y_train = []
    for i in xrange(len(y_shuffled)):
        j=y_shuffled[i]
        if d[j]<labels[j]*rate:
            y_train.append(j)
            x_train.append(x_shuffled[i])
        else:
            y_dev.append(j)
            x_dev.append(x_shuffled[i])
            d[j]-=1
    return np.array(x_train), np.array(x_dev), np.array(y_train), np.array(y_dev)

def load_test_data():
    sentence=list(open('1').readlines())
    sentence=[s.strip() for s in sentence]
    sentence=[clean_str(sent) for sent in sentence]
    sentence=[jieba.lcut(s) for s in sentence]
    return sentence