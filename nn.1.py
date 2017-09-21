# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date:   2017-09-13 14:43:03


import math
import keras
import json
import numpy as np
from pprint import pprint as p
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout


def load(datapath):
    f = open(datapath, 'r')
    data = json.load(f)
    return data['dataset'], data['labels'], dict(data['word_index'])


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
    def __init__(self, dataset, labels, wordvocab):
        self.dataset = np.array([np.array(ele) for ele in dataset])
        self.labels = np.array([np.array(ele)
                                for ele in labels])
        # self.labels = labels[0]
        self.wordvocab = wordvocab

    def trainingModel(self):
        vocabSize = len(self.wordvocab)
        embeddingDim = 2  # the vector size a word need to be converted
        maxlen = 100  # the size of a sentence vector
        outputDims = 100
        # embeddingWeights = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        hiddenDims = 100
        batchSize = 20
        train_X = self.dataset
        # Y_train = np.array([np_utils.to_categorical(
        #     ele, outputDims) for ele in self.labels])
        # Y_train = np_utils.to_categorical(self.labels, outputDims)
        # Y_trian = Y_train.reshape(-1, 100, 4)
        Y_train = np.array(self.labels)
        Y_trian = Y_train.reshape(-1, 100, 4)

        print(train_X.shape)
        print(Y_train.shape)
        # print(Y_train)
        model = Sequential()
        # , weights=[embeddingWeights]
        model.add(Embedding(output_dim=embeddingDim, input_dim=vocabSize + 1,
                            input_length=maxlen, mask_zero=True))
        model.add(LSTM(output_dim=hiddenDims, return_sequences=True))
        # model.add(LSTM(output_dim=hiddenDims, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(outputDims))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        result = model.fit(train_X, Y_train, batch_size=batchSize, epochs=20)


def main():
    dataset, labels, wordvocab = load(r'PDdatatest1.json')
    # corpus = revivification(dataset, wordvocab).reStore()
    # p(corpus)
    trainLSTM = nn(dataset, labels, wordvocab).trainingModel()


if __name__ == '__main__':
    main()
