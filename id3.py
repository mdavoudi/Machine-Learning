import math
import csv
import numpy as np
from treelib import Tree
from random import shuffle
tree = Tree()
index = 0

def loadingdata(filename, target):
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    dataset = np.array(x)
    target_data = dataset[:, target]  # class variables (class labels)
    features = np.delete(dataset, target, 1)
    return features, target_data, dataset


def classLabels(target_data):
    class_label, counts = np.unique(target_data, return_counts=True)
    return class_label, counts


def entropy(attribute):
    value, val_freqs = np.unique(attribute, return_counts=True)
    val_probs = val_freqs / len(attribute)
    return -val_probs.dot(np.log2(val_probs))


def info_gain(attribute, label):  # label is the column of labels yes no
    attr_val_counts = get_count_dict(attribute)
    total_count = len(label)
    EA = 0.0
    for attr_val, attr_val_count in attr_val_counts.items():
        EA += attr_val_count * entropy(label[attribute == attr_val])
    if math.isclose(entropy(label), EA / total_count, rel_tol=1e-5):
        return 0
    return entropy(label) - EA / total_count


def get_count_dict(data):
    data_values, data_freqs = np.unique(data, return_counts=True)
    return dict(zip(data_values, data_freqs))


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


def ID3(data, features, target_attribute_col, parent, feature_dic,a,tree):
    global index
    if len(feature_dic) == 0:
        unique = np.unique(target_attribute_col, return_counts=True)
        ind = np.argmax(unique[1])
        tree.create_node(str(a)+ ':' + str(unique[0][ind]), index, parent=parent)
        index += 1
        return tree
    elif len(data) == 0:
        return tree
    else:
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
        for a in attributes:  # high_info_col in col num. in features table
            if not pure(features, target_attribute_col, high_info_col, a):
                temp = np.take(data, np.where(features[:, high_info_col] == a), axis=0)[0]
                new_target = np.take(target_attribute_col, np.where(features[:, high_info_col] == a), axis=0)[0]
                new_feature = np.take(features, np.where(features[:, high_info_col] == a), axis=0)[0]
                ID3(temp, new_feature, new_target, pa_ind, this_child_feat,a,tree)
            else:
                new_target = np.take(target_attribute_col, np.where(features[:, high_info_col] == a), axis=0)[0]
                unic = np.unique(new_target, return_counts=True)
                ind = np.argmax(unic[1])
                tree.create_node(str(a) + ':' +str(unic[0][ind]), index, parent= pa_ind, data= str(unic[0][ind]))
                index += 1


def testing (tree, starting_node,row_data):   
    if tree[starting_node].is_leaf():
            return tree[starting_node].data
    sample=row_data[int(tree[starting_node].tag.split(':')[1])]
    for child in tree.children(starting_node):  
        if sample==child.tag.split(':')[0]:
            return testing(tree, child.identifier,row_data)
        
        
def acc(y_orig,y_predict,data):
    correctness=0
    for ind in range (len(y_orig)):
        if y_predict[ind]==y_orig[ind]:
            correctness+=1
    return correctness/len(data) *100         

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
            target_value=data[:,0]
            feature=np.delete(data,0,axis=1)
            for f in range(len(feature[0])):
                feature_dictionary[f] = np.unique(feature[:, f])
            tree=Tree()
            ID3(data, feature, target_value, -1, feature_dictionary,'root',tree)
            data_test=np.delete(test,0,axis=1)
            y_orig=test[:,0]
            y_predict=np.empty_like(y_orig)
            for row in range (len(test)):
                row_data=data_test[row,:]
                y_predict[row]=testing(tree, tree.root,row_data)
            accuracy+=acc(y_orig,y_predict,test)
        print(accuracy/(5*(j+1)))
    return accuracy
            
        
features, target_data, dataset= loadingdata('filename.csv', 0)
percission=validation(dataset)
print('%.2f' %float(percission/50))
