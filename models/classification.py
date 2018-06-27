from pyltp import Segmentor  

def cut_sentence(file_in):
    cutwords_list = [] #清零
    file_original_txt = open(file_in,'r',encoding='utf-8')
    stopwords = [ line.rstrip() for line in open('stopwords',encoding='utf-8') ] #rstrip() 删除 str末尾的指定字符（默认为空格）

    segmentor = Segmentor()
    segmentor.load('cws.model')  #加载模型
    sentences = file_original_txt.readlines()
    
    for sente in sentences:
        temp='' #用来存放被切分后的“句子”
        sente = str(sente).encode('utf-8').decode('utf-8-sig') #编码、解码。否则label会认为是 '\ufeff' 非法字符
        label = str(sente[0:1]) #去label
        temp += label+'\t'
        sente = sente[2:] #去label
        words = segmentor.segment(sente)  #分词，类型为 pyltp.VectorOfString
        word_list = list(words) #收纳在list中
        for word in word_list[1:]:
            if word not in stopwords:
                temp += word+' '
        cutwords_list.append(temp)
        
    segmentor.release()  #释放模型
    file_original_txt.close()
    return cutwords_list

# cut_sentence('train_set0.txt')

from sklearn.feature_extraction.text import TfidfTransformer #计算tfidf
from sklearn.feature_extraction.text import CountVectorizer  #计算df
vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值     

def get_tfidf(file_in):
    corpus = []
    for sente in cut_sentence(file_in):
        corpus.append(sente[2:])        #去label
#     print(corpus)
    tf=vectorizer.fit_transform(corpus) #第一个fit_transform是将文本转为词频矩阵                    
    tfidf=transformer.fit_transform(tf) #第二个fit_transform是计算tf-idf
    return tfidf      
get_tfidf('train_set0.txt')

from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import tree
from sklearn.externals import joblib
from sklearn import metrics

def training(train_set_in):
    X = get_tfidf(train_set_in)
    Y = []
    for sente in cut_sentence(train_set_in):
        Y.append(sente[0:1])
    svc=svm.SVC(C=1,kernel='poly',degree=3,gamma=10,coef0=0) #选择模型 & 参数
    lr = LogisticRegression()
    rf = RandomForestClassifier() 
    knn = KNeighborsClassifier()  
    dt = tree.DecisionTreeClassifier()
    clf = rf.fit(X,Y) #训练模型
    joblib.dump(clf,'train_model.m') #保存模型
    
def testing(test_set_in):
    file_original_txt = open(test_set_in,'r',encoding='utf-8')
    original_txt = file_original_txt.readlines() #原始文本
    i,j = 0,0
    clf = joblib.load('train_model.m') #加载模型   
    
    for sente in cut_sentence(test_set_in):
        label = int(sente[0:1])
        T = []
        T.append(str(sente[2:]))      
        tf=vectorizer.transform(T) #将数据进行转换，比如数据的归一化和标准化，将测试数据按照训练数据同样的模型进行转换，得到特征向量。
        tfidf=transformer.transform(tf)
        print(label,'\t',int(clf.predict(tfidf)),'\t',str(original_txt[j][2:])) 
        if label == int(clf.predict(tfidf)):
            i += 1 
        j += 1
    
    print('\nprecision :',i/j)
# training('train_set0.txt')
# testing('test_set0.txt')

def main():
    training('train_set0.txt')
    testing('test_set0.txt')
    
if __name__ == "__main__":  
    main()