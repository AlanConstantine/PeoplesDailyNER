# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date  : 2017-09-22 09:40:25


import json
import numpy as np
from pprint import pprint as p
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


def loadmodel(modelpath):
    model = load_model(modelpath)
    return model


def loadjson(datapath):
    f = open(datapath, 'r')
    data = json.load(f)
    return data


# '在 十五大 精神 指引 下 胜利 前进 —— 元旦 献辞 '


class pretext:
    def __init__(self, model, text, word_index):
        # super(pretext, self).__init__(*args))
        self.word_index = word_index
        self.textvec = pad_sequences(np.array([np.array(list(map(lambda word:self.word_index[word], line)))
                                               for line in text]), maxlen=100, padding='post', truncating='post', value=0)
        self.model = model

    def pre2vec(self, prossibility):
        getindex = np.argmax(prossibility)

    def getresult(self):
        predictions = self.model.predict(self.textvec)
        labelvec = [list(map(lambda pro:np.argmax(pro), prediction))
                    for prediction in predictions]
        p(labelvec)


def main():
    modelpath = r'PDmodel_epoch_100.h5'
    datapath = r'PDdata.json'
    test_X = ['在十五大精神指引下胜利前进——元旦献辞']
    model = load_model(modelpath)
    wordindex = (loadjson(datapath))['word_index']
    pre = pretext(model, test_X, wordindex).getresult()


if __name__ == '__main__':
    main()
