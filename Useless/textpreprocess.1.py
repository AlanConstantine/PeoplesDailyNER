# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date  : 2017-09-14 11:04:18
# @Python: python3.5.2


import re
import numpy as np
from pprint import pprint as p
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load(path):
    with open(path, 'r') as f:
        copurs = f.readlines()
    return corpus


class preprocess:
    def __init__(self, path):
        self.corpus = []
        self.tokenizer = None
        with open(path, 'r') as f:
            self.copurs = f.readlines()

    def word2label(self, word):
        if len(word) == 1:
            return 'S'
        elif len(word) == 2:
            return 'BE'
        elif len(word) >= 3:
            return 'B' + str((len(word) - 2) * 'M') + 'E'

    def text2vec(self):
        newscontents = []
        labels = []
        replacepunc = [']', '[', 'nt']
        for line in self.copurs[:5]:
            line = line.strip()
            newscontent = '/m'.join(line.split('/m')[1:])
            newscontent = re.sub(
                r'(?:/[a-z]{1,2})', '', newscontent)
            for punc in replacepunc:
                newscontent = newscontent.replace(punc, '')
            if len(newscontent) == 0:
                continue
            newscontents.append(newscontent.replace(' ', ''))
            newslabel = list(
                map(lambda word: self.word2label(word), newscontent.split()))
            print('text:   ', newscontent)
            print(
                '---------------------------------------------------------------------------------------')
            print('label:   ', newslabel)
            print(
                '---------------------------------------------------------------------------------------')
            break
        self.tokenizer = Tokenizer(num_words=20000, char_level=True)
        self.tokenizer.fit_on_texts(newscontents)
        # print(self.tokenizer.word_index)
        sequences = self.tokenizer.texts_to_sequences(newscontents)
        textvec = pad_sequences(sequences, maxlen=100,
                                padding='post', truncating='post')
        return textvec, self.tokenizer


def find_key(input_dict, value):
    return next((k for k, v in input_dict.items() if v == value), None)


def main():
    prpc = preprocess(r'199801_people_s_daily.txt')
    textvec, tokenizer = prpc.text2vec()
    out = []
    dic = tokenizer.word_index
    print('vec:   ', textvec[0])
    print('---------------------------------------------------------------------------------------')
    for i in textvec[0]:
        if i == 0:
            continue
        get = find_key(dic, i)
        out.append(get)
    print('restore:   ', ''.join(out))
    print('---------------------------------------------------------------------------------------')
    print('word_index_dict:   ', dic)


if __name__ == '__main__':
    main()
