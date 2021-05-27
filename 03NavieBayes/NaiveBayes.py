import numpy as np
from collections import Counter
import operator
import re
import random
import os

def loadDataSet():
    def fileName(fileDir):
        L = []
        for root, dirs, files in os.walk(fileDir):
            for file in files:
                if os.path.splitext(file)[1] == '.txt':
                    L.append(os.path.join(root, file))
        return L
    def email2list(fileName):
        email = open(fileName, 'r').read()
        listOfTokens = re.split(r'\W+', email)  # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
        return [tok.lower() for tok in listOfTokens if len(tok) > 2] #除了单个字母，例如大写的I，其它单词变成小写

    trainData,trainLabel = [],[]
    for index,folder in enumerate(['ham','spam']):
        fileDir = r'D:\PythonProject\MachineLearning\03NavieBayes\email\{}'.format(folder)
        fileList = fileName(fileDir)
        for email in fileList:
            trainData.append(email2list(email))
            trainLabel.append(index)

    # postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
    #              ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
    #              ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
    #              ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
    #              ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
    #              ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # classVec = [0,1,0,1,0,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是

    return trainData, trainLabel

def trainNavieBayes(trainData,trainLabel):
    def calcConditionProbability(trainData,trainLabel):
        def creatVocabList(dataSet):
            myVocabList = list()
            for row in dataSet:
                myVocabList.extend(row)
            return set(myVocabList)

        myVocabList = creatVocabList(trainData)
        conditionProbList = {}
        for index, data in enumerate(trainData):
            try:
                conditionProbList[trainLabel[index]]
            except:
                conditionProbList[trainLabel[index]] = {key: 0 for key in myVocabList}

            for word in data:
                conditionProbList[trainLabel[index]][word] += 1

        for label in conditionProbList.keys():
            totalNum = sum(conditionProbList[label].values()) + 2  #+2初始化为2（拉普拉斯平滑）

            for word in conditionProbList[label]:
                conditionProbList[label][word] = np.log((conditionProbList[label][word]+1) / totalNum)#+1初始化为1（拉普拉斯平滑）

        return conditionProbList

    #prior probability
    priorProb = Counter(trainLabel)
    dataSize = len(trainLabel)
    priorProb = {key:np.log(priorProb[key]/dataSize) for key in priorProb.keys()}

    #condition probability
    conditionProb = calcConditionProbability(trainData,trainLabel)

    return conditionProb,priorProb

def NaiveBayesPredict(test,conditionProb,priorProb):
    res = {}
    for key in priorProb.keys():#lnP(A|X) = lnP(X|A) + lnP(A)
        res[key] = 0
        for condition in test:

            res[key] += conditionProb[key].get(condition,0)#如果条件未出现在经验库（词汇库）则不参加计算
        res[key] += priorProb[key]

    res = sorted(res.items(),key=operator.itemgetter(1),reverse=True)
    return (res[0][0],res[0][1])

if __name__ == '__main__':
    #数据载入
    dataSet, label = loadDataSet()
    #训练集测试集划分
    testIndex = [int(random.uniform(0,len(label))) for _ in range(10)]
    testData,testLabel = [dataSet[index] for index in testIndex] , [label[index] for index in testIndex]

    temp = 0#按照顺序去掉dataSet中的测试数据，每去掉一次索引整体迁移1位
    for index in testIndex:
        del(dataSet[index - temp])
        del(label[index - temp])
        temp += 1
    trainData, trainLabel = dataSet,label
    #训练
    conditionProb,priorProb = trainNavieBayes(trainData,trainLabel)
    #测试
    # test = ['I','work']
    # res = NaiveBayesPredict(test,conditionProb,priorProb)
    # print("属于{}类概率为{}".format(res[0],np.exp(res[1])))
    correctNum,testNum = 0,len(testData)
    for test in range(testNum):
        res = NaiveBayesPredict(testData[test],conditionProb,priorProb)
        if res[0] == testLabel[test]:
            correctNum += 1
    print('预测正确率为{}%'.format(100*correctNum/testNum))