# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2018-05-24 11:03:57

import sys
sys.path.insert(0, '/Users/ronald/Documents/Alan/bello_nlp 2.0')
# sys.path.insert(0, '/Users/Allen/Downloads/Bello/bello_nlp 2.0')
# import jieba
# import jieba.posseg as pseg
from lib.hanlp import hanlp

# def jieba_seg(rawtext, stpw=True):
#     from utils import utilwords
#     stopwords = utilwords.stopwords
#     for userword in utilwords.userwords:
#         if len(userword) != 1:
#             word = userword[0]
#             jieba.add_word(word, freq=len(word) + 100, tag='nsks')
#         else:
#             word = userword[0]
#             jieba.add_word(word, freq=len(word) + 100, tag='nski')
#     seg_words = pseg.cut(rawtext)
#     token_result = jieba.tokenize(rawtext)
#     seg_result = []
#     print('Jieba segment...')
#     for w, token in zip(seg_words, token_result):
#         if stpw:
#             if str(w.word) not in stopwords:
#                 print(w.word, w.flag, token[1], token[2])
#                 seg_result.append((w.word, w.flag, token[1], token[2]))
#         else:
#             print(w.word, w.flag, token[1], token[2])
#             seg_result.append((w.word, w.flag, token[1], token[2]))
#     return seg_result


def hanlp_seg(rawtext):
    seg_result = []
    seg_tool = hanlp.nlp_tool.NLPTokenizer
    result = seg_tool.segment(rawtext)
    for n, term in enumerate(result):
        seg_result.append([n, term.word, term.nature])
    return seg_result


def nlpir_seg(rawtext):
    pass
