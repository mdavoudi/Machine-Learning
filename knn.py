import csv
import numpy as np
import operator
import math
from math import pow
from random import shuffle

def loadingdata(filename, target):
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    dataset = np.array(x)
    target_data = dataset[:, target]  # class variables (class labels)
    features = np.delete(dataset, target, 1)
    return features, target_data, dataset #, feature_dictionary

def knn(dataset, sample,k,min_distant,target_col):
    for j in range(len(dataset)):
        distance=0.0
        for i in range(len(sample)):
            if not sample[i]==dataset[j,i]:
                if type(sample[i])=="int":
                    print(">")
                    distance=distance+math.sqrt(pow(sample[i]-dataset[j,i],2))
                else:
                    distance=distance+1 
        if len(min_distant) <k:
            min_distant[j]=distance
        elif min_distant[max(min_distant.items(), key=operator.itemgetter(1))[0]] > distance:
            del min_distant[max(min_distant.items(), key=operator.itemgetter(1))[0]] 
            min_distant[j]= distance
    return vote(min_distant,target_col)
    
def vote(min_distant,target_col):
    k_nearest_y=[]
    for key in min_distant.keys():
        k_nearest_y.append(target_col[key]) 
    vote=np.unique(k_nearest_y,return_counts=True)
    return vote[0][np.argmax(vote[1])]  

def acc(y_orig,y_predict):
    correctness=0
    for ind in range (len(y_orig)):
        if y_predict[ind]==y_orig[ind]:
            correctness+=1
    return correctness/len(y_orig) *100     

           
def testing(data,test, target_col):
    y_predict=[]
    for i in range(len(test)):
        min_distant={}
        y_predict.append(knn(data, test[i,:],3,min_distant, target_col))#k=5
    return y_predict

def missing(dataset, col,label):
    target_col=dataset[:,label]
    if not col==label:
        missed=dataset[:,col]
        label=np.take(target_col,(np.where(dataset[:,col]=='?'))) #which class it is
        x=np.take(missed,(np.where(dataset[:,0]==label)[1]))
        value, freq=np.unique(x, return_counts=True)
        found=value[np.argmax(freq)]
        for index in range(len(missed)):
            if missed[index]=='?':
                dataset[index][col]=found
        return dataset, found
    else:
        colHasMissing=dataset[:,col]
        value,frequency=np.unique(colHasMissing,return_counts=True)
        found=value[np.argmax(frequency)]
        for index in range(len(colHasMissing)):
            if colHasMissing[index]=='?':
                dataset[index][col]=found
        return dataset, found
        
             
def missingInt(dataset,col,label):
     missed=dataset[:,col]
     target_col=dataset[:,label]
     label=np.take(target_col,(np.where(dataset[:,col]=='?'))) #which class it is
     x=np.take(missed,(np.where(dataset[:,0]==label)[1]))
     sum=0
     for j in range(len(x)):
         sum+=x[j]
     found=sum/len(x)
     for index in range(len(missed)):
         if missed[index]=='?':
             dataset[index][col]=found
     return dataset, found

def preprocessing(dataset,IDcol):
    prepared_data=np.delete(dataset, IDcol,axis=1)
    return prepared_data

def validation(dataset):
   accuracy=0
   for j in range(10):
        shuffle(dataset)
        test=[]
        data=[]
        split=np.array_split(dataset,5,axis=0)
        for i in range(5):       
            test=split[i]
            data=np.delete(split, i, axis=0)
            data=np.concatenate(data,axis=0)
            y_original=data[:,7]
            y_predict=np.empty_like(y_original)
            
            y_orig=test[:,7]
            test_feature=np.delete(test,7,axis=1)
            y_predict=testing(test, test_feature,y_orig)
            accuracy+=acc(y_orig,y_predict)
            print(accuracy)
   return accuracy

   
features,target_col,orig_data=loadingdata('filename.csv',7)
#dataset=preprocessing(orig_data,0)
accuracy=validation(orig_data)
print('%.2f' %float(accuracy/50))


