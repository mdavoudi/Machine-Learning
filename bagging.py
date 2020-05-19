import csv
import numpy as np
import operator
import random
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

def featureProbability(feature,classLabel,total,proDic,feature_train,data_train,col):    
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
    

def findPro_for_each_attribute_in_each_class(y_original,features,feature_train,data_train):#how many sunny,rain, high,.. in yes and No
    labelCounts=classdistribution(y_original)
    dic={}
    for c in labelCounts.keys():
        total=labelCounts[c]
        tmp_dic = {}
        for i in range(len(features[0])): #for each column gives us a dictionary
            proDic={}
            featureProbability(features[:,i],c,total,proDic,feature_train[:,i],data_train,i)
            tmp_dic[i]=proDic
        dic[c]=tmp_dic
    return dic

            
def training(row_data,target,bag_dict):
    totalPro={}
    y_in_row={}
    label=np.unique(target)
    for c in label:
        probability=1
        for j in bag_dict.keys():
            for i in range(len(row_data)):
                probability=probability*bag_dict[j][1][c][i][row_data[i]]        
            probability=probability*bag_dict[j][0][c]
            totalPro[c]=probability 
        y_in_row[j]=max(totalPro.items(), key=operator.itemgetter(1))[0]
    return y_in_row[max(y_in_row)]


        
def acc(y_orig,y_predict):
    correctness=0
    for ind in range (len(y_orig)):
        if y_predict[ind]==y_orig[ind]:
            correctness+=1
    return (correctness/len(y_orig)) *100  
    

def bagging(dataset,y_original,feature):
    bag_dict={}  
    for bag in range(10):
        array=[]
        indices=random.sample(range(len(dataset)),200)
        sample_dataset=dataset[indices,:]
        feature_train=np.delete(sample_dataset,7,axis=1)
        classPro=classProbability(sample_dataset[:,7],y_original)
        array.append(classPro)
        proDicInClass=findPro_for_each_attribute_in_each_class(y_original,feature,feature_train,sample_dataset)
        array.append(proDicInClass)
        bag_dict[bag]=array
    return bag_dict
    
        
def validation(dataset):
    accuracy=0
    for j in range(10):
        shuffle(dataset)
        y_original=dataset[:,7]
        feature=np.delete(dataset,7,axis=1)
        test=[]
        data=[]
        split=np.array_split(dataset,5,axis=0)
        for i in range(5):      
            test=split[i]
            data=np.delete(split, i, axis=0)
            data=np.concatenate(data,axis=0)
            y_train=data[:,7]
            y_predict=np.empty_like(y_train)
            bag=bagging(data,y_original,feature)
            #testing
            y_orig=test[:,7]
            test_feature=np.delete(test,7,axis=1)
            ind=0
            for row in test_feature:
                y_predict[ind]=training(row,y_orig,bag)
                ind=ind+1
            accuracy+=acc(y_orig,y_predict)
        print(accuracy/(5*(j+1)))
    return accuracy

features, target, dataset= loadingdata('filename.csv', 7)
accuracy=validation(dataset)
print('%.2f' %float(accuracy/50))
