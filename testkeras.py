# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date  : 2017-09-16 11:42:38

import keras
from pprint import pprint as p
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils.np_utils import to_categorical

import numpy as np


X = np.array([[1, 2, 3], [4, 3, 2], [6, 7, 4], [2, 1, 1], [10, 89, 90]])
Y = [[1], [2], [3], [1], [4]]
y_binary = to_categorical(Y)

print(X.shape)
# print(Y.shape)

model = Sequential()
model.add(Dense(180, input_dim=3))
model.add(Dense(100, activation='softmax'))
model.add(Dense(5, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, y_binary, epochs=200, batch_size=3)


# evaluate the model
scores = model.evaluate(X, y_binary)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

predictions = model.predict(X)
p(predictions)
# rounded = [round(x[0]) for x in predictions]
# print(rounded)

# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)

# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot

# SVG(model_to_dot(model).create(prog='dot', format='svg'))
