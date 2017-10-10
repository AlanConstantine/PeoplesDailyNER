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
# 环顾 全球 ， 日益 密切 的 世界 经济 联系 ， 日新月异 的 科技 进步 ， 正在 为 各国 经济 的 发展 提供 历史 机遇 。 但是
# ， 世界 还 不 安宁 。 南北 之间 的 贫富 差距 继续 扩大 ； 局部 冲突 时有发生 ； 不 公正 不 合理 的 旧 的 国际 政治
# 经济 秩序 还 没有 根本 改变 ； 发展中国家 在 激烈 的 国际 经济 竞争 中 仍 处于 弱势 地位 ； 人类 的 生存 与 发展 还
# 面临 种种 威胁 和 挑战 。 和平 与 发展 的 前景 是 光明 的 ， ２１ 世纪 将 是 充满 希望 的 世纪 。 但 前进 的 道路 不
# 会 也 不 可能 一帆风顺 ， 关键 是 世界 各国 人民 要 进一步 团结 起来 ， 共同 推动 早日 建立 公正 合理 的 国际 政治 经济
# 新 秩序 。


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
        print(labelvec)


def main():
    modelpath = r'PDmodel_epoch_150_batchsize_32_embeddingDim_100.h5'
    datapath = r'PDdata.json'
    test_X = ['在十五大精神指引下胜利前进——元旦献辞', '环顾全球，日益密切的世界经济联系，日新月异的科技进步，正在为各国经济的发展提供历史机遇。但是，世界还不安宁。南北之间的贫富差距继续扩大；局部冲突时有发生；不公正不合理的旧的国际政治经济秩序还没有根本改变；发展中国家在激烈的国际经济竞争中仍处于弱势地位；人类的生存与发展还面临种种威胁和挑战。和平与发展的前景是光明的，２１世纪将是充满希望的世纪。但前进的道路不会也不可能一帆风顺，关键是世界各国人民要进一步团结起来，共同推动早日建立公正合理的国际政治经济新秩序。']
    model = load_model(modelpath)
    wordindex = (loadjson(datapath))['word_index']
    pre = pretext(model, test_X, wordindex).getresult()


if __name__ == '__main__':
    main()
