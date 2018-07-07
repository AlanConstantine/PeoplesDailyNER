import json
import pickle
import random
from nerUtils import segtools
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class preparetrain:
    def __init__(self, trainpath, splitcount):
        self.tokenizer = None
        self.rawdata = []
        self.prepared_line = []
        self.prepared_label = []
        self.splitcount=splitcount
        count = 0
        with open(trainpath, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = list(json.loads(line.strip()).values())[0].replace(
                        '\r\n', '\n').replace('\t', '\n')
                    for sentence in self.clean(line):
                        self.rawdata.append(sentence)
                    count += 1
                else:
                    break
                if count % 10000 == 0:
                    print(count, 'read done...')
        self.prepare()

    def clean(self, line):
        puncs = '。;；!！？?'
        for punc in puncs:
            line = line.replace(punc, punc + '\n')
        line = line.split('\n')
        return line

    def prepare(self):
        random.shuffle(self.rawdata)
        splitnum=int(self.splitcount*len(self.rawdata))
        data = self.rawdata[:splitnum]
        print('data:',len(data))
        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(data)
        sequences = self.tokenizer.texts_to_sequences(data)
        self.prepared_line = pad_sequences(
            sequences, maxlen=50, padding='post', truncating='post', value=0)
        # create labels
        labels = []
        for line in data:
            # for sentence in line:
            label = []
            tokenized_s = [(item[1], str(item[2]))
                           for item in segtools.hanlp_seg(line)]
            for word in tokenized_s:
                if word[1] == 'nski':
                    if len(word[0]) == 1:
                        label.append(3)
                    else:
                        label.append(1)
                        for i in word[0][1:]:
                            label.append(2)
                else:
                    for i in range(len(word[0])):
                        label.append(4)
            labels.append(label)
        # random.shuffle()
        print('flag done!')
        self.prepared_label = pad_sequences(
            labels, maxlen=50, padding='post', truncating='post', value=0)

    def save2pkl(self, outputpath):
        with open(outputpath + '/traindata.json', "w") as f:
            #pickle.dump(self.prepared_line, f)
            f.write(json.dumps({'prepared_line':self.prepared_line.tolist()}))

        with open(outputpath + '/trainlabel.json', "w") as f:
            #pickle.dump(self.prepared_label, f)
            f.write(json.dumps({'prepared_label':self.prepared_label.tolist()}))

        with open(outputpath + '/word_index.json', "w") as f:
            #pickle.dump(dict(self.tokenizer.word_index), f)
            f.write(json.dumps(dict(self.tokenizer.word_index),ensure_ascii=False))


def main():
    splitcount = 0.6
    trainpath = '../cleanjd680396.json'
    outputpath = './data'
    preparetrain(trainpath, splitcount).save2pkl(outputpath)


if __name__ == '__main__':
    main()
