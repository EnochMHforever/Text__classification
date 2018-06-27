#!/usr/bin/python
# -*- coding: UTF-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
import os
from pyltp import Segmentor
from sklearn import svm
from sklearn.externals import joblib
from sklearn.decomposition import PCA


def readfile(filepath):
    '''
    读取文件并返回标签和语句
    :param filepath: 文件路径
    :return: 标签列表，句子列表
    '''
    with open(filepath, 'r',encoding='utf-8') as f:
        label = []
        source = []
        a = f.readlines()
        for i in a:
            i = i.split('\t')
            label.append(eval(i[0]))
            source.append(i[1])
    return label,source

# #test
# label,source = readfile('I:\项目\代码\samples\\test_set.txt')
# print(len(label),label)
# print(len(source),source)

def word_segment(character):
    '''
    使用pyltp分词
    :param character: 句子
    :return: 分词结果
    '''
    LTP_DATA_DIR = 'D:\TOOL\ltp_data_v3.4.0'  # ltp模型目录的路径
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    words = segmentor.segment(character)
    segment_list = list(words)
    legend = ' '.join(segment_list)
    segmentor.release()
    return legend

# #test
# legend = word_segment('厦门高雄合作优势互补厦门民进专家表示，高雄港从立法到具体实施都先行于厦门，为厦门港提供了一个很好的模板和追赶目标')
# print(legend)

def stopwordslist(filepath):
    '''
    停用词表
    :param filepath:停用词路径
    :return: 停用词表
    '''
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def vectorization(X_list):
    '''
    TF_IDF向量化语句
    :param X_list: 语句
    :return: 向量化的句子表示
    '''
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",stop_words=stopwordslist('stopwords'))
    weight = tfidf.fit_transform(X_list).toarray()
    word = tfidf.get_feature_names()
    return weight


# test
# a = ["厦门 高雄 合作 优势 互补 厦门 民进 专家 表示 ， 高雄港 从 立法 到 具体 实施 都 先行 于 厦门 ， 为 厦门港 提供 了 一个 很 好 的 模板 和 追赶 目标",
#      "没有 你 的 地方 都是 他乡","没有 你 的 旅行 都是 流浪"]
# b = vectorization(a)
# print(len(b),b)

def reduce_dim(data):
    '''
    PCA进行降维
    :param data:原矩阵
    :return: 降维后的矩阵
    '''
    pca = PCA(n_components=8000)
    newdata = pca.fit_transform(data)
    return newdata

# #test
# data = [[-1.48916667, -1.50916667],
#        [-1.58916667, -1.55916667],
#        [-1.47916667, -1.47916667],
#        [-0.48916667, -0.50916667],
#        [-0.45916667, -0.44916667],
#        [-0.50916667, -0.61916667],
#        [ 0.51083333,  0.49083333],
#        [ 0.54083333,  0.54083333],
#        [ 0.40083333,  0.59083333],
#        [ 1.51083333,  1.49083333],
#        [ 1.57083333,  1.51083333],
#        [ 1.48083333,  1.50083333]]
# newdata = reduce_dim(data)
# print(len(newdata[0]))
# print(newdata)

def model(X,Y):
    '''
    建立一个SVM模型
    :param X: 样本
    :param Y: 样本标签
    :return: 返回一个模型
    '''
    clf = svm.SVC()
    clf.fit(X,Y)
    return clf

# #test
# X = [[0, 0], [1, 1]]
# Y = [0, 1]
# model = model(X,Y)
# print(model.predict([[2., 2.]]))

def main():
    label_train,source_train = readfile('I:\项目\代码\samples\\train_set0.txt')
    label_test, source_test = readfile('I:\项目\代码\samples\\test_set0.txt')
    X_train = ''.join(source_train)
    X_test = ''.join(source_test)
    X = X_train + X_test
    # with open('final1.txt', 'a', encoding='utf-8') as f:
    #     f.write(word_segment(X))
    legend = word_segment(X)
    X = legend.split('\n')
    # print(X)
    # print(len(X))
    weight = vectorization(X)
    # weight = reduce_dim(weight)
    # print(len(weight))
    # print(weight)

    clf = model(weight[0:len(source_train)],label_train)
    joblib.dump(clf, "train_model1.m")
    clf = joblib.load('train_model1.m')
    pre = clf.predict(weight[len(source_train): ])
    sum = 0
    for i in range(len(label_test)):
        if pre[i] == label_test[i]:
            sum = sum + 1
    acc = sum/len(label_test)
    print("正确率：",acc)

if __name__ == '__main__':
    main()