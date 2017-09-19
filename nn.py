# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date:   2017-09-13 14:43:03


import math
# import keras
import json
import numpy as np
from pprint import pprint as p
# from keras.models import Sequential
# from keras.layers import Dense, Activation


def load(datapath):
    f = open(datapath, 'r')
    data = json.load(f)
    return np.array(data['dataset']), np.array(data['labels']), dict(data['word_index'])


class revivification:
    def __init__(self, dataset, word_index):
        self.dataset = dataset
        self.word_index = word_index
        self.corpus = []

    def reStore(self):
        for datum in self.dataset:
            sentence = ''.join(list(map(lambda wordindex: next((k for k, v in self.word_index.items(
            ) if v == wordindex), None), list(filter(lambda wordindex: wordindex != 0, datum)))))
            self.corpus.append(sentence)
        return self.corpus


class nn:
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def trainingModel(self):
        embeddingDim = dataset.shape[1]
        model = Sequential()
        maxlen = 100
        embeddingWeights = 0.3
        vocabSize = None
        hiddenDims = None
        batchSiz = None
        model = Sequential()
        model.add(Embedding(output_dim=embeddingDim, input_dim=vocabSize + 1,
                            input_length=maxlen, mask_zero=True, weights=[embeddingWeights]))
        model.add(LSTM(output_dim=hiddenDims, return_sequences=True))
        model.add(LSTM(output_dim=hiddenDims, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(outputDims))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        result = model.fit(train_X, Y_train, batch_size=batchSize,
                           nb_epoch=20, show_accuracy=True)


def main():
    dataset, labels, word_index = load(r'PDdataNoReduce.json')
    corpus = revivification(dataset, word_index).reStore()
    p(corpus)


if __name__ == '__main__':
    main()
