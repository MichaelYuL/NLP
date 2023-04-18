#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time   : 2023/4/15 11:08
# @Author : Lixinqian Yu
# @E-mail : yulixinqian805@gmail.com
# @File   : Project1.py
# @Project: NLP

import pandas as pd
import numpy as np
from jsonpath import jsonpath
import json
from Tookit import utils


def load_words(path):
    utils.LOG('INFO', f'Loading words from "{path}"')
    words = pd.read_excel(path, header=None)
    words_col = words.values[:, 0]
    total = words_col.shape
    utils.LOG("INFO", f"There is totally {total[0]} words!")
    return set(words_col)


def word_segment_naive(sentence):
    utils.LOG('INFO', f'WORD_SEGMENTATION')
    segments = []
    max_len = len(sentence)
    # Forward segmentation
    for i in range(1, max_len):
        temp = Forward_Segmentation(sentence, i)
        segments.append(temp) if temp not in segments else 0

    # Backward segmentation
    for i in range(1, max_len):
        temp = Backward_Segmentation(sentence, i)
        segments.append(temp) if temp not in segments else 0

    best_score = np.inf
    for seg in segments:
        score = 0
        for word in seg:
            prob = word_prob.get(word, 1e-5)
            log_prob = -np.log(prob)
            score += log_prob

        if score < best_score:
            best_score = score
            best_segment = seg

    return best_segment


def word_segment_viterbi(sentence):
    utils.LOG('INFO', 'Word segmentation by Viterbi!')
    # build DAG
    length = len(sentence)
    link_matrix = np.zeros((length+1, length+1))
    col = [i for i in range(1, length+2)]
    link_matrix = pd.DataFrame(link_matrix, index=col, columns=col)
    # one edge corresponds to one word
    # total nodes = length-1+2
    for i in range(length+1):
        for j in range(i+1, length+1):
            if word_prob.get(sentence[i:j], 0) != 0:  # existing in word_prob
                link_matrix.loc[col[i], col[j]] = -np.log(word_prob.get(sentence[i:j]))
            else:
                if sentence[i:j] in dic_words:
                    link_matrix.loc[col[i], col[j]] = -np.log(1e-5)
                else:
                    link_matrix.loc[col[i], col[j]] = np.inf
            # link_matrix.loc[col[i], col[j]] = -np.log(word_prob.get(sentence[i:j])) if word_prob.get(sentence[i:j], 0) != 0 else -np.log(1e-5)
    path_dict = {1: {"pre": 0,
                     "score": 0}}
    for i in range(2, length+2):
        pre = []
        score = np.inf
        best_node = None
        for j in range(1, i):
            if link_matrix.loc[j, i] != np.inf:
                pre.append(j)

        if not pre:
            path_dict[i] = {"pre": None,
                            "score": np.inf}
            continue

        for node in pre:
            dis = link_matrix.loc[node, i]
            total_dis = dis + path_dict[node]["score"]
            if total_dis < score:
                score = total_dis
                best_node = node

        path_dict[i] = {"pre": best_node,
                        "score": score}
    path = [length+1]
    pre = length+1
    while True:
        path.insert(0, path_dict[pre]['pre'])
        pre = path_dict[pre]['pre']
        if pre ==1:
            break
    segments = []
    for idx in range(len(path)-1):
        segments.append(sentence[path[idx]-1:path[idx+1]-1])
    return segments


def Forward_Segmentation(sentence, max_len):
    temp_segmentation = []
    while len(sentence) >= 1:
        for i in range(max_len):
            temp_sentence = sentence[0:max_len - i]
            if temp_sentence in dic_words:
                temp_segmentation.append(temp_sentence)
                sentence = sentence[max_len-i:]
                break
    return temp_segmentation


def Backward_Segmentation(sentence, max_len):
    temp_segmentation = []
    while len(sentence) >= 1:
        for i in range(max_len):
            temp_sentence = sentence[-max_len+i:]
            if temp_sentence in dic_words:
                temp_segmentation.insert(0, temp_sentence)
                sentence = sentence[:-max_len+i]
                break
    return temp_segmentation


def read_corpus(file_path):
    """
    query --> qlist = ["问题1"， “问题2”， “问题3” ....]
    answer --> alist = ["答案1", "答案2", "答案3" ....]
    务必要让每一个问题和答案对应起来（下标位置一致）
    """

    with open(file_path, 'r') as qa:
        qas = json.load(qa)
        utils.LOG("INFO", f"Already load QAs from {file_path}")

    print("Database Bersion --> "+jsonpath(qas, "$.version")[0])
    qlist, alist = [], []
    for section in jsonpath(qas, "$.data")[0]:
        for paragraph in jsonpath(section, "$.paragraphs")[0]:
            for qa in jsonpath(paragraph, "$.qas")[0]:
                if not jsonpath(qa, "$.answers[0].text"):
                    continue
                query = jsonpath(qa, "$.question")[0]
                answer = jsonpath(qa, "$.answers[0].text")[0]
                qlist.append(query)
                alist.append(answer)

    assert len(qlist) == len(alist)  # 确保长度一样
    print(len(qlist))
    return qlist, alist


if __name__ == '__main__':
    '''dic_path = './data/综合类中文词库.xlsx'
    # TODO : step-1 load chinese words from dic.txt
    dic_words = load_words(dic_path)
    utils.LOG('INFO', 'Loading...DONE!')

    # 以下是每一个单词出现的概率。为了问题的简化，我们只列出了一小部分单词的概率。 在这里没有出现的的单词但是出现在词典里的，统一把概率设置成为0.00001
    # 比如 p("学院")=p("概率")=...0.00001
    word_prob = {"北京": 0.03, "的": 0.08, "天": 0.005, "气": 0.005, "天气": 0.06, "真": 0.04, "好": 0.05, "真好": 0.04, "啊": 0.01,
                 "真好啊": 0.02,
                 "今": 0.01, "今天": 0.07, "课程": 0.06, "内容": 0.06, "有": 0.05, "很": 0.03, "很有": 0.04, "意思": 0.06,
                 "有意思": 0.005, "课": 0.01,
                 "程": 0.005, "经常": 0.08, "意见": 0.08, "意": 0.01, "见": 0.005, "有意见": 0.02, "分歧": 0.04, "分": 0.02,
                 "歧": 0.005}
    # TODO : step-2-1 segment the sentences
    utils.LOG("TEST", 'Test word segmentation!')
    print(word_segment_naive("北京的天气真好啊"))
    print(word_segment_naive("今天的课程内容很有意思"))
    print(word_segment_naive("经常有意见分歧"))
    utils.LOG("TEST", 'Test Done!')
    # TODO : step-2-2 segment the sentences by Viterbi
    utils.LOG("TEST", 'Test Viterbi!')
    print(word_segment_viterbi("北京的天气真好啊"))
    print(word_segment_viterbi("今天的课程内容很有意思"))
    print(word_segment_viterbi("经常有意见分歧"))
    utils.LOG("TEST", 'Test Done!')'''
    # TODO : step-3-1 load QAs
    utils.LOG("INFO", "Build FQA system!")
    utils.LOG("INFO", "Load question-answer pairs...")
    qlist, alist = read_corpus("./data/train-v2.0.json")
    utils.LOG("INFO", "Loading Done")
