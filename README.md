# test_classification
## 1.Introduction
   **Function** : This project constructs a model to implement text classification through different machine learning algorithms.<br><br>
   **Requirement** : python 3

## 2.corpus
  We use the----ACE corpus, which is commonly used in event extraction, as our original dataset.
<div align=center><img width="554.8" height="200" src="https://github.com/qwjaskzxl/event_classification/blob/master/image/ace%20corpus.png" alt="ace corpus"/></div>

The 9 types of corresponding labels are as follows:

|Notanyclass|	Life|Movement|Transaction|Business|Conflict|Contact	|Personnel|Justice|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|1|2|3|4|5|6|7|8|
<p align="center">type dictionary</p>
   
## 3.Extract samples
[out.txt](:storage\3cb00c28-f19b-4703-bfdb-baa843b33176\ec4b2bcc.txt) 
   66 articles were selected as the test set, the remaining 567 as training set, and 33 articles selected randomly from the training set as the validation set.
   
   **dataset**：
   [train_set](https://github.com/qwjaskzxl/text-classification/blob/master/samples/train_set.txt) ／
   [test_set](https://github.com/qwjaskzxl/text-classification/blob/master/samples/test_set.txt) ／
   [validation set](https://github.com/qwjaskzxl/text-classification/blob/master/samples/ver_set.txt)
   
    The way is to read the XML file from "etree", find the corresponding tag by "Find", and "XPath" return the content that needs to be tagged.
## 4.Pre Processing
  1. text processing as the format required by the model. <br>
  2. participle. 3. go to the discontinuation of the word.
Participle: use the CWS segmentation model of Pyltp to process the whole text set and return a list of finished words.
Get stop words: expand the scope of the stop words, and remove the strange words that appear only once in the whole training set.
  //code：[c.py](:storage\7baa3ef0-d75e-4c64-bedc-f451dda79824\43150200.py)
  //预处理的结果：[build_set.txt](:storage\3cb00c28-f19b-4703-bfdb-baa843b33176\cad4251d.txt)

## 5.Feature Engineering
### 文本变向量、特征处理、特征选择、特征降维
	文本变向量：TF_IDF，TF是样本的词频，IDF是逆文档频率，一个词语出现的词向量由这两个数据决定，
	TF越大说明在样本中的权重越大，IDF越大说明在整个文档中出现的次数都是巨大的，权重会随之减少。
	特征降维：PCA
代码：[predict.py](:storage\7baa3ef0-d75e-4c64-bedc-f451dda79824\f95c4f76.py)

## 6.Models
### SVM
### RandowForest

## 7.Evaluation

## 8.Optimization
	调参方法:由于经验不足，采取网格搜索的方法进行地毯式搜索
	具体：在python中建立笛卡尔积列表，将两种超参数进行组合，然后在模型中，选择一个拟合分数最好的超平面系数。
	同时打印每个输出的结果，人工进行分析比较，最终得出C=[2-5],gamma=[0.2,0.3,0.4]中的效果明显好于其他组合，
	也达到了可以与线性核函数媲美的效果，说明整体调参思路正确。
	分析：C>1，说明模型对于错误样本的惩罚会高，对于训练集的拟合效果会更好，为什么泛化效果也优，可能是因为我们的
	测试机与训练集来自于同一文本，gamma大小偏小，说明每个样本对于超平面的影响是比较大的。

