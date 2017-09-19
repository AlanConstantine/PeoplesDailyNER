# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date  : 2017-09-14 11:04:18
# @Python: python3.5.2


import re
import json
# import collections
import numpy as np
from pprint import pprint as p
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class preprocess:
    def __init__(self, path):
        self.corpus = []
        self.tokenizer = None
        with open(path, 'r') as f:
            self.copurs = f.readlines()
        # labelnum = collections.OrderedDict({'B': 1, 'M': 2, 'E': 3, 'S': 4})

    def voc2label(self, word):
        if len(word) == 1:
            # return 'S'
            return '4'
        elif len(word) == 2:
            # return 'BE'
            return '13'
        elif len(word) >= 3:
            # return 'B' + str((len(word) - 2) * 'M') + 'E'
            return '1' + str((len(word) - 2) * '2') + '3'

    def text2vec(self):
        newscontents = []
        labels = []
        replacepunc = [']', '[', 'nt']
        splitpunc = ['。', '？', '！', '；']
        for line in self.copurs:
            newsparagraphs = '/m'.join((line.strip()).split('/m')[1:])
            # replace tags.
            newsparagraphs = re.sub(
                r'(?:/[a-zA-Z]{1,2})', '', newsparagraphs)
            # replace the mark punctuation.
            for rpunc in replacepunc:
                newsparagraphs = newsparagraphs.replace(rpunc, '')
            # replace the punctuation which can split text to sentences.
            for spunc in splitpunc:
                newsparagraphs = newsparagraphs.replace(spunc, spunc + '/')
            # split the text by '/'.
            for newscontent in newsparagraphs.split('/'):
                replablock = newscontent.replace(' ', '')
                if len(replablock) == 0:
                    continue
                newscontents.append(replablock)

                # get news label
                newslabel = ''.join(list(map(lambda word: self.voc2label(word), list(
                    filter(lambda x: len(x) != 0, newscontent.split())))))
                # if len(newslabel) < 100:
                # newslabel = newslabel + 'N' * (100 - len(newslabel))
                # newslabel = newslabel + '0' * (100 - len(newslabel))
                # elif len(newslabel) > 100:
                # newslabel = newslabel[:100]
                # labels.append(list(newslabel))
                labels.append(list(map(int, newslabel)))

        self.tokenizer = Tokenizer(char_level=True)
        # self.tokenizer = Tokenizer(num_words=20000, char_level=True)
        self.tokenizer.fit_on_texts(newscontents)
        sequences = self.tokenizer.texts_to_sequences(newscontents)
        # textvec = pad_sequences(sequences)
        textvec = pad_sequences(sequences, maxlen=100,
                                padding='post', truncating='post', value=0)
        return textvec, np.array(labels), dict(self.tokenizer.word_index)

    def save2json(self, savepath):
        textvec, labels, word_index = self.text2vec()
        savedict = {'dataset': textvec.tolist(), 'labels':
                    labels.tolist(), 'word_index': word_index}
        with open(savepath, 'w') as f:
            f.write(json.dumps(savedict))
        return "save done."


def output(path, content):
    with open(path, 'a') as f:
        f.write(content)
    return '%s done!' % path


def outvec(path, vecs):
    outlist = []
    for vec in vecs:
        outlist.append(str(list(filter(lambda ele: ele != 0, vec))))
    print(output(path, '\n'.join(outlist)))


def main():
    corpus = r'199801_people_s_daily.txt'
    savepath = r'PDdata.json'
    # textvec, labels, tokenizer = preprocess(corpus).text2vec()
    # print(output(r'vocab.txt', str(tokenizer)))
    # outvec(r'text2vec.txt', textvec.tolist())
    # print(textvec.shape)
    # print(labels.shape)
    print(preprocess(corpus).save2json(savepath))


if __name__ == '__main__':
    main()
