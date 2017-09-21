# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date:   2017-09-13 14:43:03


import json
import keras
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
        self.dataset = np.array([np.array(ele) for ele in dataset])
        self.labels = np.array([np.array(ele)
                                for ele in labels])
        self.wordvocab = wordvocab

    def trainingModel(self):
        vocabSize = len(self.wordvocab)
        embeddingDim = 2  # the vector size a word need to be converted
        maxlen = 100  # the size of a sentence vector
        outputDims = 4
        # embeddingWeights = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        hiddenDims = 100
        batchSize = 20
        train_X = self.dataset
        Y_train = np.array([np_utils.to_categorical(
            ele, outputDims) for ele in self.labels])

        print(train_X.shape)
        print(Y_train.shape)
        max_features = vocabSize + 1
        word_input = Input(shape=(maxlen,), dtype='float32', name='word_input')
        mask = Masking(mask_value=0.)(word_input)
        word_emb = Embedding(max_features, embeddingDim,
                             input_length=maxlen, name='word_emb')(mask)
        bilstm1 = Bidirectional(
            LSTM(hiddenDims, return_sequences=True))(word_emb)
        #bilstm2 = Bidirectional(LSTM(hiddenDims, return_sequences=True))(bilstm1)
        bilstm_d = Dropout(0.5)(bilstm1)
        output = TimeDistributed(
            Dense(outputDims, activation='softmax'))(bilstm_d)
        model = Model(inputs=[word_input], outputs=output)
        #sgd = optimizers.SGD(lr=0.1, decay=1e-3)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'], )
        # plot_model(model, to_file='model.png',
        #    show_layer_names=True, show_shapes=True)

        # , weights=[embeddingWeights]
        # model.add(Embedding(output_dim=embeddingDim, input_dim=vocabSize + 1,
        #                     input_length=maxlen, mask_zero=True))
        # model.add(LSTM(output_dim=hiddenDims, return_sequences=True))
        # model.add(LSTM(output_dim=hiddenDims, return_sequences=False))
        # model.add(Dropout(0.5))
        # model.add(Dense(outputDims))
        # model.add(Activation('softmax'))
        # model.compile(loss='categorical_crossentropy',
        #               optimizer='adam', metrics=['accuracy'])

        result = model.fit(train_X, Y_train, batch_size=batchSize, epochs=20)
        model.save('PDmodel.h5')


def main():
    dataset, labels, wordvocab = load(r'PDdatatest1.json')
    # corpus = revivification(dataset, wordvocab).reStore()
    # p(corpus)
    trainLSTM = nn(dataset, labels, wordvocab).trainingModel()


if __name__ == '__main__':
    main()
