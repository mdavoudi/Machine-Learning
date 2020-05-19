import math
import csv
import numpy as np
from treelib import Tree
import random
from random import shuffle

index = 0

def loadingdata(filename, target):
    feature_dictionary = {}
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    dataset = np.array(x)
    target_data = dataset[:, target]
    features = np.delete(dataset, target, 1)
    for f in range(len(features[0])):
        feature_dictionary[f] = np.unique(features[:, f])
    return features, target_data, dataset,feature_dictionary

def info_gain(attribute, label):
    counts =np.unique(attribute, return_counts=True)
    total_count = len(label)
    EA = 0.0
    value=counts[0]
    key=counts[1]
    for i in range(len(counts[0])):
        EA += key[i] * entropy(label[attribute == value[i]])
    if math.isclose(entropy(label), EA / total_count, rel_tol=1e-5):
        return 0
    return entropy(label) - EA / total_count

def find_highest_info_gain(feature, class_var,feature_dict):
    best_feature_num = 0
    best_feature_val = 0.0
    for attr_num in range(len(feature[0])):
        if attr_num in feature_dict:
            info_gain_val = info_gain(feature[:, attr_num], class_var)
            if info_gain_val >= best_feature_val:
                best_feature_num = attr_num
                best_feature_val = info_gain_val
    return best_feature_num

def pure(feature, target_col, parent_ind, child_name):
    result = np.reshape(target_col[np.where(feature[:, parent_ind] == child_name)], (-1, 1))
    if np.all(result == result[:, 0]):
        return True
    else:
        return False

def entropy(attribute):
    value, value_freqs = np.unique(attribute, return_counts=True)
    val_probs = value_freqs / len(attribute)
    return -val_probs.dot(np.log2(val_probs))

def random_tree_learning(data,Stree,target_attribute_col,parent,features,a,feature_dict):
    global index    
    if len(data)==0:
        return Stree    
    elif len(feature_dict) < 2:
        unique = np.unique(target_attribute_col, return_counts=True)
        ind = np.argmax(unique[1])
        Stree.create_node(str(a)+ ':' + str(unique[0][ind]), index, parent=parent, data= str(unique[0][ind]))
        index += 1
        return Stree
    elif len(feature_dict.keys())<2:
        unique = np.unique(target_attribute_col, return_counts=True)
        ind = np.argmax(unique[1])
        Stree.create_node(str(a)+ ':' + str(unique[0][ind]), index, parent=parent, data= str(unique[0][ind]))
        index += 1
        return Stree
    elif len(features) <2:
        return Stree
    else:
        random_index=random.sample(feature_dict.keys(),2)
        feature_subset=features[:,random_index]
        high_info_col = find_highest_info_gain(feature_subset, target_attribute_col,feature_dict)
        if parent == -1:
            Stree.create_node(str(a) +':' + str(random_index[high_info_col]), index, data= str(random_index[high_info_col]))
            pa_ind = index 
            index += 1
        else:
            Stree.create_node(str(a) +':' + str(random_index[high_info_col]), index, parent=parent, data= str(random_index[high_info_col]))
            pa_ind = index
            index += 1         
        this_child_feature=feature_dict.copy()
        del this_child_feature [random_index[high_info_col]]
        attributes = np.unique(features[:, random_index[high_info_col]])
        for a in attributes:  
            if not pure(features, target_attribute_col, random_index[high_info_col], a):
                new_data = np.take(data, np.where(features[:, random_index[high_info_col]] == a), axis=0)[0]
                new_target = np.take(target_attribute_col, np.where(features[:, random_index[high_info_col]] == a), axis=0)[0]
                new_feature = np.take(features, np.where(features[:, random_index[high_info_col]] == a), axis=0)[0]
                random_tree_learning(new_data,Stree,new_target,pa_ind, new_feature,a,this_child_feature) 
            else:
                new_target = np.take(target_attribute_col, np.where(features[:, random_index[high_info_col]] == a), axis=0)[0]
                unic = np.unique(new_target, return_counts=True)
                ind = np.argmax(unic[1])
                Stree.create_node(str(a) + ':' +str(unic[0][ind]), index, parent= pa_ind)
                index += 1

def random_forest(data,tree,orig_data,parent,a,feature_dict):
    global index
    data_subset=np.ndarray(shape=(800,len(orig_data[0])),dtype=str)    
    indices=np.random.choice(len(orig_data),size=800)
    data_subset=orig_data[indices]
    features=np.delete(data_subset,0,1)
    target_col=data_subset[:,0]
    random_tree_learning(data_subset,tree,target_col,parent,features,a,feature_dict)
    return tree

def testing (data,tree, starting_node,row_data,y_predict,target_col):   
    if tree[starting_node].is_leaf():
            return str(tree[starting_node].tag.split(':')[1])
    flag=0   
    sample=row_data[int(tree[starting_node].tag.split(':')[1])]
    for child in tree.children(starting_node):  
        if sample==child.tag.split(':')[0]:
            flag=1
            return testing(data,tree, child.identifier,row_data, y_predict,target_col)
    if flag==0:
        unique = np.unique(data[:,int(tree[starting_node].tag.split(':')[1])], return_counts=True)
        ind = np.argmax(unique[1]) 
        most_common=np.take(target_col, np.where(data[:,int(tree[starting_node].data)] == unique[0][ind]), axis=0)[0]
        count=np.unique(most_common,return_counts=True)
        return count[0][np.argmax(count[1])]

def acc(y_orig,y_predict):
    correctness=0
    for ind in range (len(y_orig)):
        if y_predict[ind]==y_orig[ind]:
            correctness+=1
    return correctness/len(y_orig) *100


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


def validation(dataset):
    accuracy=0
    for j in range(10):
        shuffle(dataset)
        test=[]
        data=[]
        split=np.array_split(dataset,5,axis=0)
        for i in range(5): 
            feature_dictionary={}
            test=split[i]
            data=np.delete(split, i, axis=0)
            data=np.concatenate(data,axis=0)
            feature=np.delete(data,0,axis=1)       
            for f in range(len(feature[0])):
                feature_dictionary[f] = np.unique(feature[:, f])
            for k in range(10):     
                array_trees=[]
                tree=Tree()
                array_trees.append(random_forest(data,tree,data,-1,'Root',feature_dictionary)) 
            y_orig=test[:,0]
            y_predict=np.empty_like(y_orig)
            test_feature=np.delete(test,0,axis=1)
            for row in range (len(test_feature)): 
                y_pre=[]
                row_data=test[row,:]
                for t in range(len(array_trees)):
                    y_pre.append(str(testing(test,array_trees[t], array_trees[t].root,row_data,y_predict,y_orig)))
                value,frequency=np.unique(y_pre,return_counts=True)
                y_predict[row]=value[np.argmax(frequency)]
            accuracy+=acc(y_orig,y_predict)
        print(accuracy/(5*(j+1)))
    return accuracy
   
features,target_col,orig_data,feature_dict=loadingdata('filename.csv',0)
percission=validation(orig_data)
print('%.2f' %float(percission/50))