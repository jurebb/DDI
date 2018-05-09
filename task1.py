import nltk 
import xml.etree.ElementTree as ET 
import os
import pandas as pd
import numpy as np
from xlm_parsers_functions import *
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score


__PARENT_PATH = '../Train/DrugBank/'
__CHEMSPOT_PATH = '../molminer_vir/task1_1_chemspot_test.txt'


def dfFromXML():
    headers = [
        'sentence_id', 
        'sentence_text', 
        'entity_id', 
        'entity_name', 
        'entity_charOffset', 
        'entity_type'
              ]
    
    entities_dataset = []
    parent_directory = __PARENT_PATH
    for filename in os.listdir(parent_directory):
        if filename.endswith(".xml"):
            tree = ET.parse(parent_directory + filename)
            entities_dataset = entities_dataset + listEntitiesFromXML(tree.getroot())
    
    df = pd.DataFrame(entities_dataset, columns=headers)
    return df

def load_from_file(filename):
    with open(filename, 'r') as f:
            content = f.readlines()
    
    past_id = content[0][6:-1]
    print(past_id)
    
    dataset = []
    sentence_data = []
    entities = []
    
    counter = 0
    for line in content:
        if(line.startswith('#sid#')):
            if(counter > 0):
                dataset.append([sentence_data, entities])
            sentence_data = []
            sentence_data.append(line[6:-1])
            entities = []
            
        elif(line.startswith('#s#')):
            sentence_data.append(line[4:-1])
        else:
            entities.append(line[:-1].split('|'))
        counter += 1
    dataset.append([sentence_data, entities])
    
    return dataset

def write_to_file():
    with open('Exported.txt', 'w') as file:
        for ent in range(len(entities_dataset)):
            file.write(entities_dataset[ent][1] + '\n')
         
def construct_dataset(daf):
    diagnostic = False
    
    df3 = pd.DataFrame(daf.copy())
    del df3['entity_name']
    df3 = df3.drop_duplicates().as_matrix()
    
    print(daf.shape)
    print(df3.shape)
    
    daf = daf.as_matrix()
    
    print('construct_dataset(): new_data:')
    new_data = []
    temp = []
    for i in range(len(df3)):
        text = nltk.word_tokenize(df3[i][1])
        tagged_sent = nltk.pos_tag(text)
        
        comp = []
        for d in daf:
            if(d[0] == df3[i][0]):
               cc = nltk.word_tokenize(d[2])
               for c in cc:
                    comp.append(c)
                    
        for t in tagged_sent:
            ##find the same daf
            
            if(t[0] in comp):
                temp = (t[0], t[1], df3[i][0], df3[i][1], 'DRUG')
            else: 
                temp = (t[0], t[1], df3[i][0], df3[i][1], 'None')
            new_data.append(temp)
        if(diagnostic):
            if(i>0 and i%58 == 0):
                print(i, 'of', len(df3))
                print('diagnostic:')
                print('text', text)
                print('comp', comp)
                print('new_data[-1]', new_data[-1])
                print('new_data[-2]', new_data[-2])
                print('new_data[-3]', new_data[-3])
                print('new_data[-4]', new_data[-4])
                print('new_data[-5]', new_data[-5])
                print('new_data[-6]', new_data[-6])
                print('new_data[-7]', new_data[-7])
                print('new_data[-8]', new_data[-8])
                print('new_data[-9]', new_data[-9])
                print('new_data[-10]', new_data[-10])
                break
        else:
            if(i>0 and i%508 == 0):
                print(i, 'of', len(df3))
    return new_data
        
    
def task1_1_nltk(df):    
    print("======================== task 1_1 =============================")
    
    del df['entity_charOffset']
    del df['entity_id']
    del df['entity_type']
    
    print(df.shape)
    
    df2 = df
    
    data = construct_dataset(df2)
    
    headers2 = [
        'token_name',
        'token_tag',
        'sentence_id',
        'sentence_text', 
        'entity_name'
            ]
    
    df2 = pd.DataFrame(data, columns=headers2)
    
    df_train, df_test = train_test_split(df2, test_size = 0.2, random_state=22, shuffle = False)
    
    text_train = df_train['sentence_text'].as_matrix()
    text_test = df_test['sentence_text'].as_matrix()
    
    print('text_train.shape', text_train.shape)
    
    
    sw = stopwords.words("english")
    vectorizer = TfidfVectorizer(lowercase=True, binary = True, stop_words=sw, sublinear_tf  = True, norm = None)
    
    x_train = vectorizer.fit_transform(text_train).toarray()
    x_test = vectorizer.transform(text_test).toarray()
    token_name_train = vectorizer.transform(df_train['token_name'].as_matrix()).toarray()
    token_name_test = vectorizer.transform(df_test['token_name'].as_matrix()).toarray()
    
#this is an attempt to concatenate token tags to the dataset, memory leak problems
    #token_name_train = np.column_stack((token_name_train, df_train['token_tag']))
    #token_name_test = np.column_stack((token_name_test, df_test['token_tag']))
    #[:,None]
    
    x_train = np.concatenate((x_train, token_name_train), axis=1)
    x_test = np.concatenate((x_test, token_name_test), axis=1)
     
    print('x_train.shape', x_train.shape)
    
    y_train = df_train['entity_name'].as_matrix()
    y_test = df_test['entity_name'].as_matrix()
    
    print('y_train.shape', y_train.shape)
    
    svc = LinearSVC(C =0.0004, class_weight = 'balanced')
    svc.fit(x_train, y_train)
    pred = svc.predict(x_test)
    
    
    print(pred)
    print(accuracy_score(pred, y_test))
    print(f1_score(pred, y_test, average = 'macro'))
    
    
def task1_2(df):
    print("======================== task 1_2 =============================")
    df_train, df_test = train_test_split(df, test_size = 0.2, shuffle = False)

    print(df_train.shape)
    
    del df['sentence_id']
    del df['entity_id']
    del df['entity_charOffset']
    
    print(df_train.shape)
    print(df.head())
    
    
    text_train = df_train['sentence_text'].as_matrix()
    text_test = df_test['sentence_text'].as_matrix()
    
    sw = stopwords.words("english")
    vectorizer = TfidfVectorizer(lowercase=True, binary = True, stop_words=sw, sublinear_tf  = True, norm = None)
    
    x_train = vectorizer.fit_transform(text_train).toarray()
    x_test = vectorizer.transform(text_test).toarray()
    
    entity_name_train = vectorizer.transform(df_train['entity_name'].as_matrix()).toarray()
    entity_name_test = vectorizer.transform(df_test['entity_name'].as_matrix()).toarray()

    x_train = np.concatenate((x_train, entity_name_train), axis=1)
    x_test = np.concatenate((x_test, entity_name_test), axis=1)
   
    y_train = df_train['entity_type'].as_matrix()
    y_test = df_test['entity_type'].as_matrix()
    
    svc = LinearSVC(C =0.0004, class_weight = 'balanced')
    svc.fit(x_train, y_train)
    pred = svc.predict(x_test)
    
    
    print(pred)
    print(accuracy_score(pred, y_test))
    print(f1_score(pred, y_test, average = 'macro'))
    
    preds = list(set(pred))
    br = []
    
    for j in range(len(preds)):
        br.append(0)
        
    for i in pred:
        for j in range(len(preds)):
            if i == preds[j]:
                br[j] += 1
    print(br)
    
    
    preds = list(set(y_test))
    br = []
    
    for j in range(len(preds)):
        br.append(0)
        
    for i in y_test:
        for j in range(len(preds)):
            if i == preds[j]:
                br[j] += 1
    print(br)
         
    
    
if __name__=="__main__":
    df = dfFromXML()
    
    #a = load_from_file(__CHEMSPOT_PATH)
    data = task1_1_nltk(df)
    #task1_2(df)
