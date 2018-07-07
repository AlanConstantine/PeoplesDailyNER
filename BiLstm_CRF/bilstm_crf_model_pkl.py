# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2018-06-21 16:46:54

import json
import keras
import pickle
import numpy as np
from datetime import datetime as dt
from keras.utils import np_utils
from keras_contrib.layers import CRF
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, Bidirectional, Input, Masking, TimeDistributed


class LoadData(object):
    def __init__(self):
        self.traindata = self.loadpkl('./data/traindata.pkl')
        self.trainlabel = self.loadpkl('./data/trainlabel.pkl')
        self.word_index = self.loadpkl('./data/word_index.pkl')

    def loadpkl(self, pklpath):
        with open(pklpath, 'rb') as inp:
            data = pickle.load(inp)
        return data

    def getdata(self):
        return self.traindata, self.trainlabel, self.word_index


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
        self.dataset = dataset
        self.labels = labels
        self.wordvocab = dict(wordvocab)

    def trainingModel(self):
        vocabSize = len(self.wordvocab)
        embeddingDim = 100  # the vector size a word need to be converted
        maxlen = 50  # the size of a sentence vector
        outputDims = 4 + 1
        hiddenDims = 100
        batchSize = 100
        # NUM_CLASS = 4

        X_train = self.dataset
        Y_train = np_utils.to_categorical(self.labels, outputDims)

        print('vocabSize:', vocabSize)
        print('traindata.shape:', X_train.shape)
        print('label.shape:', Y_train.shape)

        max_features = vocabSize + 1

        word_input = Input(
            shape=(maxlen, ), dtype='float32', name='word_input')
        mask = Masking(mask_value=0.)(word_input)
        word_emb = Embedding(
            max_features, embeddingDim, input_length=maxlen,
            name='word_emb')(mask)
        bilstm1 = Bidirectional(LSTM(hiddenDims,
                                     return_sequences=True))(word_emb)
        bilstm2 = Bidirectional(
            LSTM(hiddenDims, return_sequences=True))(bilstm1)
        bilstm_d = Dropout(0.5)(bilstm2)

        dense = TimeDistributed(Dense(outputDims,
                                      activation='softmax'))(bilstm_d)

        crf_layer = CRF(outputDims, sparse_target=False)
        crf = crf_layer(dense)
        model = Model(inputs=[word_input], outputs=[crf])
        model.summary()

        model.compile(
            optimizer='adam',
            loss=crf_layer.loss_function,
            metrics=[crf_layer.accuracy])

        result = model.fit(X_train, Y_train, batch_size=batchSize, epochs=150)

        model.save(
            'bilstm-crf_epoch_150_batchsize_100_new.h5')

    def save2json(self, json_string, savepath):
        with open(savepath, 'w', encoding='utf8') as f:
            f.write(json_string)
        return "save done."


def main():
    traindata, trainlabel, word_index = LoadData().getdata()
    trainLSTM = nn(traindata, trainlabel, word_index).trainingModel()


if __name__ == '__main__':
    ts = dt.now()
    main()
    te = dt.now()
    spent = te - ts
    print('[Finished in %s]' % spent)
