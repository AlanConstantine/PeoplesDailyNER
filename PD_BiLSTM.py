# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date:   2017-09-13 14:43:03


import json
import keras
from datetime import datetime as dt
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, Bidirectional, Input, Masking, TimeDistributed


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
        self.dataset = np.array(dataset)
        self.labels = np.array(labels)
        self.wordvocab = wordvocab

    def trainingModel(self):
        vocabSize = len(self.wordvocab)
        embeddingDim = 100  # the vector size a word need to be converted
        maxlen = 100  # the size of a sentence vector
        outputDims = 4 + 1
        hiddenDims = 100
        batchSize = 32
        train_X = self.dataset
        train_Y = np_utils.to_categorical(self.labels, outputDims)

        print(train_X.shape)
        print(train_Y.shape)
        max_features = vocabSize + 1
        word_input = Input(shape=(maxlen,), dtype='float32', name='word_input')
        mask = Masking(mask_value=0.)(word_input)
        word_emb = Embedding(max_features, embeddingDim,
                             input_length=maxlen, name='word_emb')(mask)
        bilstm1 = Bidirectional(
            LSTM(hiddenDims, return_sequences=True))(word_emb)
        bilstm2 = Bidirectional(
            LSTM(hiddenDims, return_sequences=True))(bilstm1)
        bilstm_d = Dropout(0.5)(bilstm2)
        output = TimeDistributed(
            Dense(outputDims, activation='softmax'))(bilstm_d)
        model = Model(inputs=[word_input], outputs=output)
        #sgd = optimizers.SGD(lr=0.1, decay=1e-3)
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'], )

        result = model.fit(train_X, train_Y, batch_size=batchSize, epochs=10)
        model.save('PDmodel_epoch_10_batchsize_32_embeddingDim_100_new2.h5')

    def save2json(self, json_string, savepath):
        with open(savepath, 'w', encoding='utf8') as f:
            f.write(json_string)
        return "save done."


def main():
    dataset, labels, wordvocab = load(r'PDdata.json')
    # corpus = revivification(dataset, wordvocab).reStore()
    trainLSTM = nn(dataset, labels, wordvocab).trainingModel()


if __name__ == '__main__':
    ts = dt.now()
    main()
    te = dt.now()
    spent = te - ts
    print('[Finished in %s]' % spent)
