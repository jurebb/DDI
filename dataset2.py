import nltk 
import xml.etree.ElementTree as ET 
import os
import pandas as pd
import numpy as np
from xlm_parsers_functions import *

from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score



def none_dataSet(df):
    #pitat babana jel dobro
    headers = [
        'sentence_id', 
        'sentence_text', 
        'entity_id', 
        'entity_name1', 
        'entity_charOffset', 
        'entity_type1'
            ]

    entities_dataset = []
    parent_directory = 'semeval_task9_train\\Train\\DrugBank\\'
    for filename in os.listdir(parent_directory):
        if filename.endswith(".xml"):
            tree = ET.parse(parent_directory + filename)
            entities_dataset = entities_dataset + listEntitiesFromXML(tree.getroot())

    df2 = pd.DataFrame(entities_dataset, columns=headers)

    print(df2.head())
    #babanu dosta
    del df2['entity_charOffset']

    data1 = []
    curr_sentence_id  = ''
    temp = []
    for d1 in df2.as_matrix():
        if d1[1] != curr_sentence_id:
            if len(temp) != 0:
                data1.append(temp)
            temp = [(d1[1], d1[3])]
            curr_sentence_id = d1[1]
        else:
            temp.append((d1[1], d1[3]))

    print('ajmoooo')
    print(len(data1))
    for i in range(3):
        print(len(data1[i]))
        print(data1[i])

    data = []

    for i in range(len(data1)):
        if len(data1[i]) > 1:
            for j in range(len(data1[i])):
                for k in range(j + 1, len(data1[i])):
                    data.append([data1[i][j][0], data1[i][j][1], data1[i][k][1], 'None']) #dodajemo tuple (recenica, entitet_j, entitet_k, 'None')


    print('vussssssss')
    print(len(data))
    for i in range(3):
        print(len(data[i]))
        print(data[i])

    for i in range(len(data)):
        for dd in df.as_matrix():
            d = data[i]
            if (dd[2] == d[2] and dd[1] == d[1]) or (dd[2] == d[1] and dd[1] == d[2]):
                data[i][3] = dd[3]
                break


    return data


def main():

    headers = [
        'sentence_id', 
        'sentence_text', 
        'entity1_id', 
        'entity1_name', 
        'entity1_type',
        'entity2_id', 
        'entity2_name', 
        'entity2_type',
        'interection_type'
            ]

    entities_dataset = []
    parent_directory = 'semeval_task9_train\\Train\\DrugBank\\'
    for filename in os.listdir(parent_directory):
        if filename.endswith(".xml"):
            tree = ET.parse(parent_directory + filename)
            entities_dataset = entities_dataset + listDDIFromXML(tree.getroot())


    df = pd.DataFrame(entities_dataset, columns=headers)

    del df['sentence_id']
    del df['entity1_id']
    del df['entity2_id']

    del df['entity1_type']
    del df['entity2_type']

    print(df.shape)

    headers = [
        'sentence_text', 
        'entity1_name', 
        'entity2_name', 
        'interection_type'
            ]

    #print(df.head())
    
    data = none_dataSet(df)

    df = pd.DataFrame(data, columns=headers)
    
    df_train, df_test = train_test_split(df, test_size = 0.2, shuffle = False)

    print(df_train.head())

    print(df_train.shape)

    text_train = df_train['sentence_text'].as_matrix()
    text_test = df_test['sentence_text'].as_matrix()

    sw = stopwords.words("english")
    vectorizer = TfidfVectorizer(lowercase=True, stop_words = sw, binary = True, sublinear_tf  = True, norm = None)

    x_train = vectorizer.fit_transform(text_train).toarray()
    x_test = vectorizer.transform(text_test).toarray()

    print(x_train.shape)

    entity1_name_train = vectorizer.transform(df_train['entity1_name'].as_matrix()).toarray()
    entity1_name_test = vectorizer.transform(df_test['entity1_name'].as_matrix()).toarray()

    entity2_name_train = vectorizer.transform(df_train['entity2_name'].as_matrix()).toarray()
    entity2_name_test = vectorizer.transform(df_test['entity2_name'].as_matrix()).toarray()

    x_train = np.concatenate((x_train, entity1_name_train), axis=1)
    x_test = np.concatenate((x_test, entity1_name_test), axis=1)

    x_train = np.concatenate((x_train, entity2_name_train), axis=1)
    x_test = np.concatenate((x_test, entity2_name_test), axis=1)

    print(x_train.shape)

    y_train = df_train['interection_type'].as_matrix()
    y_test = df_test['interection_type'].as_matrix()

    '''
    Mozemo radit kompromis izmedu toga koliko nan je stalo do odredene klase, acc i f1
    class_w_dict = dict()
    class_w_dict['None'] = 0.1
    diff_values = list(set(y_test)) 
    for v in diff_values:
        class_w_dict[v] = 40
    '''

    lgr = LogisticRegression(C = 0.0004, class_weight = 'balanced')
    lgr.fit(x_train, y_train)
    pred = lgr.predict(x_test)

    svc = LinearSVC(C = 0.0004, class_weight = 'balanced')
    svc.fit(x_train, y_train)
    pred = svc.predict(x_test)

    rfc = ensemble.RandomForestClassifier(class_weight = 'balanced')
    rfc.fit(x_train, y_train)
    pred = rfc.predict(x_test)

    gb = ensemble.GradientBoostingClassifier()
    gb.fit(x_train, y_train)
    pred = gb.predict(x_test)

    br = 0
    for p in pred:
        if p != pred[0]:
            br += 1
    print(br / len(pred))
    print(pred)
    print(accuracy_score(pred, y_test))
    print(f1_score(pred, y_test, average = 'macro'))

main()