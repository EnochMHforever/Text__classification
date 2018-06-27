from pyltp import Segmentor
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer  # 计算tfidf
from sklearn.feature_extraction.text import CountVectorizer  # 计算df
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import itertools

segmentor = Segmentor() # 加载模型
X_train = []
y_train = []
X_text = []
y_text = []
stopwords = []

def get_text_and_label(file_in):
    text_list, label_list = [], []
    with open(file_in, encoding='utf-8') as f:
        for T in f.readlines():
            sentence = str(T[2:]).encode('utf-8').decode('utf-8-sig').rstrip()
            words = segmentor.segment(sentence)  # 分词，类型为 pyltp.VectorOfString
            words = ' '.join(list(words))
            text_list.append(words)
            label_list.append(int(T[0:1]))
    return text_list, label_list


def model(train_set_in, test_set_in,arg1 = 1,arg2 = 2,arg3 = 1,arg4 = 4):
    X_train, y_train = get_text_and_label(train_set_in)
    X_test, y_test = get_text_and_label(test_set_in)
    #     for i in range(10):
    #         print(y_test[i],X_test[i])
    step = [('vect', CountVectorizer(stop_words=stopwords)),
            ('tfidf', TfidfTransformer()),
            #             ('pca',PCA(n_components=3000)),
            # ('clf', RandomForestClassifier()),
            # ('clf',svm.SVC(C = arg1,kernel='poly', degree=arg2, gamma=arg3, coef0=arg4,probability=True))
            # ('clf', svm.SVC(C=arg1, kernel='linear', degree=arg2, gamma=arg3, coef0=arg4, probability=True))
            ('clf', svm.SVC(C=arg1, kernel='rbf', degree=arg2, gamma=arg3, coef0=arg4, probability=True))
            ]

    ppl_clf = Pipeline(step)
    ppl_clf.fit(X_train, y_train)
    probability = ppl_clf.predict_proba(X_test)
    #     for p in probability:

    #     print(size(probability))# 输出分类概率

    joblib.dump(ppl_clf, 'train_model.m')  # 保存模型
    #     ppl_clf = joblib.load('train_model.m') #加载模型

    prediction = ppl_clf.predict(X_test)
    precision = np.mean(prediction == y_test)  # 准确率
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(y_test, prediction))  # 混淆矩阵


#     svc=svm.SVC(C=1,kernel='poly',degree=3,gamma=10,coef0=0) #选择模型 & 参数
#     lr = LogisticRegression()
#     rf = RandomForestClassifier()
#     knn = KNeighborsClassifier()
#     dt = tree.DecisionTreeClassifier()
#     ? MultinomialNB()

def main():
    segmentor.load('D:\TOOL\ltp_data_v3.4.0\cws.model')
    stopwords = [line.rstrip() for line in open('stopwords', encoding='utf-8')]  # rstrip() 删除 str末尾的指定字符（默认为空格）
    print("模型评估调C：")
    c = [0.4,0.6,0.8,1,2,2.5,3,3.5,4,4.5,5,10]
    gamma = [0.1,0.2,0.3,0.4]
    cobim = []
    for i in itertools.product(c, gamma):
        cobim.append(i)
    print(cobim)
    for i in cobim:
        print("c = ",i[0]," ","gamma = ",i[1])
        model("I:/项目/代码/samples/train_set0.txt", "I:/项目/代码/samples/test_set0.txt",arg1 = i[0],arg3 = i[1])
        print()
    # model("I:/项目/代码/samples/train_set0.txt", "I:/项目/代码/samples/test_set0.txt",arg1=5,arg3=0.4)
    segmentor.release()


if __name__ == "__main__":
    main()