#!/usr/bin/env python
# coding: utf-8

# In[153]:



from collections import Counter
import numpy as np
import pandas as pd
import sklearn.model_selection as cross
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
##Please install tqdm library to launch the progress bar.


# In[154]:


train_set = pd.read_excel("EPL_Training.xlsx")
test_set = pd.read_excel("EPL_Testing.xlsx")


# In[155]:


def eucliden_distance (point1 , point2 ):
    '''Calculate the eucliden distance'''
    return np.linalg.norm(point1 - point2)


# In[156]:


def classification_accuracy(actual, predicted):
    correct = sum(actual[i] == predicted[i] for i in range(0,len(actual)))
    x = len(predicted) - correct
    y = len(actual) - x
    percent = y / len(actual)
    return percent * 100


# In[157]:


def most_frequent(list):
    counter = 0
    num = list[0]
    for i in list:
        current_frequency = list.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


# In[158]:




def get_error(predicted, actual):
    correct = sum((predicted[i] == actual[i]) for i in range(len(predicted)))
    return len(predicted) - correct


# In[159]:


def get_suitable_k():
    k_info = {}
    best_k_records = []
    for _ in tqdm(range(1000)):
        least_error = 1000  # same as no of iterations
        best_k = 1
        new_train, new_test = cross.train_test_split(train_set, test_size=0.05)
        only_needed_new_test = new_test[
            ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC"]
        ]
        result_test_column = new_test["FTR"].to_numpy()
        possible_k_values = int(len(new_train) / 3)
        for k in range(1, possible_k_values):
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            only_needed_new_train = new_train[
                ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC"]
            ]
            result_train_column = new_train["FTR"]
            knn_classifier.fit(only_needed_new_train, result_train_column)
            classifier_prediction = knn_classifier.predict(only_needed_new_test)

            error = get_error(classifier_prediction, result_test_column)

            if k in k_info:
                current_k = k_info[k]
                current_k["count_k"] = current_k["count_k"] + 1
                current_k["sum_errors"] = current_k["sum_errors"] + error
                k_info[k] = current_k
            else:
                k_info[k] = {"count_k": 1, "sum_errors": error}

            if error < least_error:
                least_error = error
                best_k = k

        best_k_records.append(best_k)
    plot_average_k(k_info)
    x=np.array(best_k_records)
    return np.bincount(x).argmax()
    ##c = Counter(best_k_records)
    ##return c.most_common(1)[0][0]

        


# In[160]:


def classification(best_k):
    modified_test_set = test_set[["HS","AS","HST","AST","HF","AF","HC","AC"]]
    modified_train_set = train_set[["HS","AS","HST","AST","HF","AF","HC","AC"]]
    train_column = train_set["FTR"]
    knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
    knn_classifier.fit(modified_train_set, train_column)
    test_column=test_set["FTR"].to_numpy()
    prediction=knn_classifier.predict(modified_test_set)
    correct = len(test_column) - get_error(prediction, test_column)
    accuracy_precent=correct/len(test_column)
    return accuracy_precent*100
    
    
    
    

    


# In[161]:


def plot_average_k(k_dict):
    x = []
    y = []
    for key in k_dict:
        x.append(key)
        y.append(k_dict[key]["sum_errors"] / k_dict[key]["count_k"])
    df = pd.DataFrame({"key": x, "average": y})
    df.plot(x="key", y="average")


# In[162]:


best_k = get_suitable_k()
accuracy=classification(best_k)
print("The accuracy is", accuracy,"%")

##the accuracy in 63.9% ,, please run the code to figure it
##I tried to find a way to make a progress bar ,, As the code takes time to run ,So I want to make your waiting nice :)


# In[ ]:




