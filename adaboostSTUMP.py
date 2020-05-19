
import csv
import operator
import numpy as np
from treelib import Tree
from random import shuffle
index = 0

def loadingdata(filename, target):
    feature_dictionary = {}
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    dataset = np.array(x)
    target_data = dataset[:, target]  # class variables (class labels)
    features = np.delete(dataset, target, 1)
    for f in range(len(features[0])):
        feature_dictionary[f] = np.unique(features[:, f])
    return features, target_data, dataset, feature_dictionary


def classLabels(target_data):
    class_label, counts = np.unique(target_data, return_counts=True)
    return class_label, counts


def entropy(attribute):
    value, val_freqs = np.unique(attribute, return_counts=True)
    val_probs = val_freqs / len(attribute)
    return -val_probs.dot(np.log2(val_probs))


def info_gain(attribute, label):  # label is the column of labels yes no
    counts =np.unique(attribute, return_counts=True)
    total_count = len(label)
    EA = 0.0
    value=counts[0]
    key=counts[1]
    for i in range(len(counts[0])):
        EA += key[i] * entropy(label[attribute == value[i]])
    return entropy(label) - EA / total_count

def find_highest_info_gain(feature, class_var, feature_dic_copy):
    best_feature_num = 0.0
    best_feature_val = 0.0
    for attr_num in range(len(feature[0])):
        if attr_num in feature_dic_copy:
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

def stump(data, features, target_attribute_col, parent, feature_dic,a,tree):
    global index
    high_info_col = find_highest_info_gain(features, target_attribute_col, feature_dic)
    if parent == -1:
        tree.create_node(str(a) +':' + str(high_info_col), index,data= str(high_info_col))
        pa_ind = index 
        index += 1
    else:
        tree.create_node(str(a) +':' + str(high_info_col), index, parent=parent, data= str(high_info_col))
        pa_ind = index
        index += 1
    this_child_feat = feature_dic.copy()
    del this_child_feat[high_info_col]
    attributes = np.unique(features[:, high_info_col])        
    for a in attributes:
        makechild(data, high_info_col, features, target_attribute_col, parent, feature_dic,a,tree, pa_ind)


def makechild(data, high_info_col, features, target_attribute_col, parent, feature_dic,a,tree, pa_ind):
    global index
    new_target = np.take(target_attribute_col, np.where(features[:, high_info_col] == a), axis=0)[0]
    unic = np.unique(new_target, return_counts=True)
    ind = np.argmax(unic[1])
    tree.create_node(str(a) + ':' +str(unic[0][ind]), index, parent= pa_ind, data= str(unic[0][ind]))
    index += 1
    
def adaboost_testing(tree, starting_node, row_data, data_origin,target_col):
    if tree[starting_node].is_leaf():
            return tree[starting_node].data
    flag=0
    sample=row_data[int(tree[starting_node].tag.split(':')[1])]
    for child in tree.children(starting_node):  
        if sample==child.tag.split(':')[0]:
            flag=1
            return adaboost_testing(tree, child.identifier,row_data, data_origin,target_col)
    if flag==0:
        unique = np.unique(data_origin[:,int(tree[starting_node].tag.split(':')[1])], return_counts=True)
        ind = np.argmax(unique[1]) 
        most_common=np.take(target_col, np.where(data_origin[:,int(tree[starting_node].data)] == unique[0][ind]), axis=0)[0]
        count=np.unique(most_common,return_counts=True)
        return count[0][np.argmax(count[1])]
        

def adaboost_learning(data,weights,trees_list):
    if len(trees_list)<5:
        error=0.0
        vote_rate=0.0
        total_sum=0.0
        feature_dictionary = {}
        indices=np.random.choice(len(data), 2000, replace=True, p=weights)
        random_data=data[indices]
        tree=Tree()
        target=random_data[:,0]  
        features=np.delete(random_data,0,1)
        for f in range(len(features[0])):
            feature_dictionary[f] = np.unique(features[:, f])    
        stump(random_data, features, target, -1, feature_dictionary,'root',tree)
        y_original=data[:,0]
        y_predict=np.empty_like(y_original)
        for row in range (len(data)):
            row_data=data[row,:]
            y_predict[row]=adaboost_testing(tree,tree.root,row_data,data, y_original) 
            if not y_predict[row]== y_original[row]:
                error= error+weights[row]
        if error==0:
            trees_list[vote_rate]=tree
            return trees_list
        vote_rate=float(error/(1-error))
        for index in range(len(y_original)):
            if y_original[index]==y_predict[index]:
                weights[index]=weights[index]*vote_rate
        for w in weights:
            total_sum=total_sum+w
        weights= [x/total_sum for x in weights]        
        trees_list[vote_rate]=tree
        adaboost_learning(data, weights,trees_list)

def acc(y_orig,y_predict):
    correctness=0
    for ind in range (len(y_orig)):
        if y_predict[ind]==y_orig[ind]:
            correctness+=1
    return (correctness/len(y_orig)) *100 

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
            test=split[i]
            data=np.delete(split, i, axis=0)
            data=np.concatenate(data,axis=0)
            weights=[]
            trees_list={}
            weights +=len(data)*[1/len(data)]# weights size = length data & weights' elements = 1/length data
            adaboost_learning(data, weights,trees_list)
            y_original=test[:,0]
            y_pre=np.empty_like(y_original)
            for row in range (len(test)):
                row_data=test[row,:]
                result={}
                for voting,tree in trees_list.items():
                    result[adaboost_testing(tree,tree.root,row_data,test, y_original)]=voting  
                highest_voiting_rate=max(result.items(), key=operator.itemgetter(1))[0]
                y_pre[row]=highest_voiting_rate
            accuracy+=acc(y_original,y_pre)
        print(accuracy/(5*(j+1)))
    return accuracy
        
        
feature, target_data, dataset, feature_dict = loadingdata('filename.csv', 0)
percission=validation(dataset)
print('%.2f' %float(percission/50))

