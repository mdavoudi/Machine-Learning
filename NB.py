import csv
import numpy as np
import operator
from random import shuffle
index = 0
def loadingdata(filename, target):
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    dataset = np.array(x)
    target_data = dataset[:, target]  # class variables (class labels)
    features = np.delete(dataset, target, 1)
    return features, target_data, dataset #, feature_dictionary


def classdistribution(target_data):
    labelCounts={}
    unic=np.unique(target_data, return_counts=True)
    key=unic[0]
    value=unic[1]  
    for ind in range(len(unic[0])):
        labelCounts[key[ind]]=value[ind]
    return labelCounts


def classProbability(y_train, y_original):
    labelProbability={}
    unic=np.unique(y_original, return_counts=True)
    key=unic[0] 
    for i in range(len(key)):
        labelProbability[key[i]]=0
    unic_train=np.unique(y_train, return_counts=True)
    key_train=unic_train[0]
    value=unic_train[1]
    for ind in range(len(key_train)):
        labelProbability[key[ind]]=value[ind]/len(y_train)
    return labelProbability

def featureProbability(feature,classLabel,total,proDic,feature_train,data_train):    
    attributes=np.unique(feature)
    for a in attributes:
        data=np.take(data_train,np.where(feature_train==a),axis=0)[0]
        if len(data)== 0:
            proDic[a]=0
        else:
            filtered_data=np.take(data_train,np.where(data==classLabel),axis=0)[0]
            probability=len(filtered_data)/total
            proDic[a]=probability
    return proDic
    

def findPro_for_each_attribute_in_each_class(y_original,features,feature_train,data_train):
    labelCounts=classdistribution(y_original)
    dic={}
    for c in labelCounts.keys():
        total=labelCounts[c]
        tmp_dic = {}
        for i in range(len(features[0])): #for each column gives us a dictionary
            proDic={}
            featureProbability(features[:,i],c,total,proDic,feature_train[:,i],data_train)
            tmp_dic[i]=proDic
        dic[c]=tmp_dic
    return dic

def findPro_NoMatter_class(dataset,features):
    pro_dic={}
    total_data=len(dataset)
    for i in range(len(features[0])):
        attributes=np.unique(features[:,i])
        for a in attributes:
            a_count=len(np.take(features[:,i],np.where(features[:,i]==a),axis=0)[0])
            pro_dic[a]=a_count/total_data
    return pro_dic
            
def testing(row_data,target,classPro,proDicInClass):
    totalPro={}
    label=np.unique(target)
    for c in label:
        probability=1
        for i in range(len(row_data)):
            probability=probability*proDicInClass[c][i][row_data[i]]        
        probability=probability*classPro[c]
        totalPro[c]=probability
    return max(totalPro.items(), key=operator.itemgetter(1))[0]
        
def acc(y_orig,y_predict,data):
    correctness=0
    for ind in range (len(y_orig)):
        if y_predict[ind]==y_orig[ind]:
            correctness+=1
    return (correctness/len(data)) *100   
    
def validation(dataset):
    accuracy=0
    for j in range(10):
        shuffle(dataset)
        y_original=dataset[:,0]
        feature=np.delete(dataset,0,axis=1)
        test=[]
        data=[]
        split=np.array_split(dataset,5,axis=0)
        for i in range(5): 
            test=split[i]
            data=np.delete(split, i, axis=0)
            data=np.concatenate(data,axis=0)
            y_train=data[:,0]
            feature_train=np.delete(data,0,axis=1)
            y_predict=np.empty_like(y_train)
            classPro=classProbability(y_train,y_original)
            proDicInClass=findPro_for_each_attribute_in_each_class(y_original,feature,feature_train,data)
            #Testing
            y_orig=test[:,0]
            test_feature=np.delete(test,0,axis=1)
            ind=0
            for row in test_feature:
                y_predict[ind]=testing(row,y_orig,classPro,proDicInClass)
                ind=ind+1
            accuracy+=acc(y_orig,y_predict,test)
        print(accuracy/(5*(j+1)))
    return accuracy

features, target, dataset= loadingdata('filename.csv', 0)
accuracy=validation(dataset)
print('%.2f' %float(accuracy/50))


