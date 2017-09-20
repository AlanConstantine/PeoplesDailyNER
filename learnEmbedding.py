# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date  : 2017-09-19 14:57:51

from keras.preprocessing.text import Tokenizer


# tokenizer = Tokenizer()
# texts = ["The sun is shining in June!", "September is grey.",
#          "Life is beautiful in August.", "I like it", "This and other things?"]
# tokenizer.fit_on_texts(texts)
# print(tokenizer.word_index)
# tokenizer.texts_to_sequences(["June is beautiful and I like it!"])


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

# X = tokenizer.texts_to_matrix(texts)
# y = [1, 0, 0, 0, 0]

# vocab_size = len(tokenizer.word_index) + 1

# model = Sequential()
# model.add(Dense(2, input_dim=vocab_size))
# model.add(Dense(1, activation='sigmoid'))


# model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# model.fit(X, y=y, batch_size=200, nb_epoch=700, verbose=0,
#           validation_split=0.2, shuffle=True)

# from keras.utils.np_utils import np as np
# res = np.round(model.predict(X))
# print(res)
import numpy as np

X = np.array([[0.,  1.,  1.,  1.],
              [0.,  1.,  0.,  0.],
              [0.,  1.,  1.,  0.],
              [0.,  0.,  0.,  0.]])

model = Sequential()
model.add(Embedding(5, 2, input_length=4))
model.add(Flatten())
model.compile('rmsprop', 'mse')
output_array = model.predict(X)
print(output_array)
