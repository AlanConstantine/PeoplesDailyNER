# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date:   2017-09-15 10:47:16

from pprint import pprint as p
import os
import sys
from keras.preprocessing.text import Tokenizer
BASE_DIR = ''
# GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + './20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)

# print('Found %s texts.' % len(texts))
# p(texts[1])
# print('====')
# p(texts[2])
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=200)
tokenizer.fit_on_texts(texts)
print(tokenizer.word_index)
# sequences = tokenizer.texts_to_sequences(texts)
# data = pad_sequences(sequences, maxlen=10)
# print(data)
