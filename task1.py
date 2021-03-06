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
from scipy.sparse import dia_matrix
import gc
from tempfile import mkdtemp
import os.path as path
from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors

__PARENT_PATH = 'semeval_task9_train\\Train\\MedLine\\'
#__PARENT_PATH = '../Train/DrugBank/'
__CHEMSPOT_PATH = '../molminer_vir/task1_1_chemspot_test.txt'


def simple_nn(x_train, x_test, y_train, n_classes = 4):

    print(x_train.shape)

    model = Sequential()

    model.add(Dense(100, input_dim = x_train.shape[1]))
    model.add(Activation('relu'))

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    adam = Adam(lr = 0.01)

    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['accuracy']) 

    model.fit(x_train, y_train, nb_epoch = 13, batch_size = 100)

    pred = model.predict_classes(x_test, 100) #pred_temp je formata [[1], [0.5], [-0.7], ...] umjesto normalnog niza predikcija [1, 0.5, -0.7, ...] 

    return pred


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

def construct_dataset_tutto(daf,daf_with_entity_type):
    diagnostic = False

    del df['entity_type']
    
    df3 = pd.DataFrame(daf.copy())
    del df3['entity_name']
    df3 = df3.drop_duplicates().as_matrix()
    
    print(daf.shape)
    print(df3.shape)
    print('aaa')
    
    
    daf = daf.as_matrix()
    print(daf[0])
    
    print('construct_dataset_tutto(): new_data:')
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
            temp_type = '0'
            if(t[0] in comp):
                #print(daf_with_entity_type.head())
                for index, d in daf_with_entity_type.iterrows():
                    #print(t[1], d['sentence_id'])
                    if (df3[i][0] == d['sentence_id']):
                        #print('printprinttutto', t[0].lower(),'-',  d['entity_name'].lower())    
                        if t[0].lower() in d['entity_name'].lower():
                            temp_type = d['entity_type']
                            #print('hit')
                            break
                    else:
                        continue
                      #token, #pos_tag, #sent id, #sent text, #
                temp = (t[0], t[1], df3[i][0], df3[i][1], temp_type)
            else: 
                temp = (t[0], t[1], df3[i][0], df3[i][1], 'None')
            new_data.append(temp)
        if(diagnostic):
            if(i>0 and i%58 == 0):
                print(i, 'of', len(df3))
                print('diagnostic:')
                print('text', text)
                print('comp', comp)
                print('new_data[-31]', new_data[-171])
                print('new_data[-32]', new_data[-172])
                print('new_data[-33]', new_data[-173])
                print('new_data[-34]', new_data[-174])
                print('new_data[-35]', new_data[-175])
                print('new_data[-36]', new_data[-176])
                print('new_data[-37]', new_data[-177])
                print('new_data[-38]', new_data[-178])
                print('new_data[-39]', new_data[-179])
                print('new_data[-40]', new_data[-180])
                print('new_data[-41]', new_data[-181])
                print('new_data[-12]', new_data[-182])
                print('new_data[-13]', new_data[-183])
                print('new_data[-13]', new_data[-184])
                print('new_data[-13]', new_data[-185])
                print('new_data[-13]', new_data[-186])
                print('new_data[-13]', new_data[-187])
                print('new_data[-13]', new_data[-188])
                print('new_data[-13]', new_data[-189])
                print('new_data[-13]', new_data[-190])
                print('new_data[-13]', new_data[-191])
                break
            
        else:
            if(i>0 and i%58 == 0):
                print(i, 'of', len(df3))
    return new_data
        
def task_1_tuttocompleto(df):
    print("======================== task_1_tuttocompleto =============================")
    del df['entity_charOffset']
    del df['entity_id']
    
    print(df.shape)
    
    df2 = df.copy(deep=True)
    
    data = construct_dataset_tutto(df, df2)
    
    print('bitno', df.shape)  #df nema izbrisan entity type
    
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
     

    del sw
    del vectorizer
    del token_name_train
    del token_name_test
    del df2
    del data
    #return x_train, x_test, y_train, y_test, df_train['token_tag'], df_test['token_tag']
    y_train = df_train['entity_name'].astype("category").cat.codes.as_matrix()
    y_test = df_test['entity_name'].astype("category").cat.codes.as_matrix()

    lb = LabelBinarizer()

    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    
    pred = simple_nn(x_train, x_test, y_train, 5)

    #pred = lb.inverse_transform(pred)
    y_train = lb.inverse_transform(y_train)
    y_test = lb.inverse_transform(y_test)
 
    pred_list = [pred]

    print(accuracy_score(pred, y_test))
    print(f1_score(pred, y_test, average = 'macro'))

    lgr = LogisticRegression(C = 0.05, class_weight = 'balanced')
    lgr.fit(x_train, y_train)
    pred1 = lgr.predict(x_test)

    pred_list.append(pred1)

    print(accuracy_score(pred1, y_test))
    print(f1_score(pred1, y_test, average = 'macro'))

    svc = LinearSVC(C = 0.0004, class_weight = 'balanced')
    svc.fit(x_train, y_train)
    pred2 = svc.predict(x_test)

    pred_list.append(pred2)

    print(accuracy_score(pred2, y_test))
    print(f1_score(pred2, y_test, average = 'macro'))


    #gb = ensemble.GradientBoostingClassifier()
    #gb.fit(x_train, y_train)
    #pred = gb.predict(x_test)

    #pred_list.append(pred)

    final_pred = []
    for i in range(len(pred_list[0])):
        temp = [0, 0, 0, 0, 0]
        for j in range(len(pred_list)):
            temp[pred_list[j][i]] += 1

        final_pred.append(np.argmax(temp))
    
    pred = final_pred

    print(pred)
    print(accuracy_score(pred, y_test))
    print(f1_score(pred, y_test, average = 'macro'))

    np.save('my_pred', pred)
    np.save('y_test', y_test)

    joblib.dump(lb, 'lb')
    
    
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

    id_test = df_test['sentence_id'].as_matrix()
    
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
    
    del sw
    del vectorizer
    del token_name_train
    del token_name_test
    del df2
    del data
    
    #return x_train, x_test, y_train, y_test, df_train['token_tag'], df_test['token_tag']
    svc = LinearSVC(C =0.0004, class_weight = 'balanced')
    svc.fit(x_train, y_train)
    pred = svc.predict(x_test)

    np.save('my_pred', pred)
    np.save('y_test', y_test)
    np.save('sentence_id', id_test)
    
    
    print(pred)
    print(accuracy_score(pred, y_test))
    print(f1_score(pred, y_test, average = 'macro'))
    
def task1_1_with_tags(x_train, x_test, y_train, y_test, df_train_tag, df_test_tag):
    
    gc.collect()
    
#the sparse matrix approach, untested, unfinished
# =============================================================================
#     target_shape = []
#     target_shape[0] = x_train.shape[0]
#     target_shape[1] = x_train.shape[1] + 1
#     d = dia_matrix(target_shape, dtype=np.int8).toarray()
#     print('sparse.shape', d.shape)
# =============================================================================
    filename1 = path.join(mkdtemp(), 'train1.dat')
    filename2 = path.join(mkdtemp(), 'test1.dat')
    print('filename1', filename1)
    print('filename2', filename2)
    target_shape = []
    target_shape.append(x_train.shape[0])
    target_shape.append(x_train.shape[1] + 1)
    
    fp_train = np.memmap(filename1, mode='w+', shape=tuple(target_shape))
    fp_train[:,:-1] = x_train[:]
    del x_train
    
    #map tags to ints 
    tags = df_train_tag.unique()
    tags_dict = dict(zip(tags, range(len(tags))))
    df_train_tag = df_train_tag.replace(tags_dict)
    df_test_tag = df_test_tag.replace(tags_dict)
    fp_train[:, -1] = df_train_tag[:]
    del df_train_tag
    #del fp_train
    
    target_shape = []
    target_shape.append(x_test.shape[0])
    target_shape.append(x_test.shape[1] + 1)
    
    fp_test = np.memmap(filename2, mode='w+', shape=tuple(target_shape))
    fp_test[:,:-1] = x_test[:]
    del x_test
    fp_test[:, -1] = df_test_tag[:]
    del df_test_tag
    #del fp_test
    
    filenamey1 = path.join(mkdtemp(), 'ytrain.dat')
    filenamey2 = path.join(mkdtemp(), 'ytest.dat')
    print('filenamey1', filenamey1)
    print('filenamey2', filenamey2)
    fp_ytrain = np.memmap(filenamey1, mode='w+', dtype="S10", shape=(85725, 1))
    fp_ytest = np.memmap(filenamey2, mode='w+', dtype="S10", shape=(21432, 1))
    y_train=y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    fp_ytrain[:] = y_train[:]
    fp_ytest[:] = y_test[:]
    del fp_ytrain
    del fp_ytest
# =============================================================================
#     x_train = np.column_stack((x_train, df_train_tag))
#     x_test = np.column_stack((x_test, df_test_tag))
#     
#     svc = LinearSVC(C =0.0004, class_weight = 'balanced')
#     svc.fit(x_train, y_train)
#     pred = svc.predict(x_test)
# =============================================================================
    
    print('fptrain shape', fp_train.shape)
    print('fptest shape', fp_test.shape)
    del fp_train
    del fp_test
    
    svc = LinearSVC(C =0.0004, class_weight = 'balanced')
    svc.fit(fp_train, y_train)
    pred = svc.predict(fp_test)
    
    print('tags, x_train.shape', x_train.shape)
    print('tags, y_train.shape', y_train.shape)
    
    print(pred)
    print(accuracy_score(pred, y_test))
    print(f1_score(pred, y_test, average = 'macro'))
    
def task1_1_load_train():
    newfp1 = np.memmap('/tmp/tmpiwx41mac/train1.dat', mode='r+', shape=(85725, 9745))
    newfp2 = np.memmap('/tmp/tmp7ymrni1b/test1.dat', mode='r+', shape=(21432, 9745))
    
    y_train = np.memmap('/tmp/tmpmk21p80a/ytrain.dat', mode='r+',dtype="S10",  shape=(85725, 1))
    y_test = np.memmap('/tmp/tmptl_8upsq/ytest.dat', mode='r+', dtype="S10", shape=(21432, 1))
    
    svc = LinearSVC(C =0.0004, class_weight = 'balanced')
    svc.fit(newfp1, y_train)
    pred = svc.predict(newfp2)
    
    print('tags, x_train.shape', newfp1.shape)
    print('tags, y_train.shape', y_train.shape)
    
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
    
    print(df.head())
    new_data = []
    all_tags = []
    for i in range(len(df_train)):
        text = nltk.word_tokenize(df_train['sentence_text'][i])
        tagged_sent = nltk.pos_tag(text)
        found = False
        for t in tagged_sent:
            if(t[0].lower() in df_train['entity_name'][i].lower() and found==False):
                new_data.append(t[1])
                all_tags.append(t[1])
                found=True
        if(found==False):
            new_data.append('0')  
            all_tags.append('0')
    dftrain_tag = pd.DataFrame(new_data, columns=['entity_tag'])
    #print(x_train.shape)
    #print(len(new_data))
    
    new_datat = []
    for i in range(10343, 10343 + len(df_test)):
        text = nltk.word_tokenize(df_test['sentence_text'][i])
        tagged_sent = nltk.pos_tag(text)
        found = False
        for t in tagged_sent:
            if(t[0].lower() in df_test['entity_name'][i].lower() and found==False):
                new_datat.append(t[1])
                all_tags.append(t[1])
                found=True
        if(found==False):
            new_datat.append('0')
            all_tags.append('0')
    dftest_tag = pd.DataFrame(new_datat, columns=['entity_tag'])
    
    df_alltags = pd.DataFrame(all_tags, columns=['entity_tag'])
    #print(x_test.shape)
    #print(len(new_datat))
    #print(dftrain_tag.head())
    tags = df_alltags['entity_tag'].unique()
    tags_dict = dict(zip(tags, range(len(tags))))
    dftrain_tag = dftrain_tag.replace(tags_dict)
    dftest_tag = dftest_tag.replace(tags_dict)
    #print(dftrain_tag.head())
    
    x_train = np.concatenate((x_train, dftrain_tag), axis=1)
    x_test = np.concatenate((x_test, dftest_tag), axis=1)
    #print(x_train[0:6])
    
    y_train = df_train['entity_type'].astype("category").cat.codes.as_matrix()
    y_test = df_test['entity_type'].astype("category").cat.codes.as_matrix()

    lb = LabelBinarizer()

    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    
    pred = simple_nn(x_train, x_test, y_train)

    #pred = lb.inverse_transform(pred)
    y_train = lb.inverse_transform(y_train)
    y_test = lb.inverse_transform(y_test)
 
    pred_list = [pred]

    print(accuracy_score(pred, y_test))
    print(f1_score(pred, y_test, average = 'macro'))

    lgr = LogisticRegression(C = 0.05, class_weight = 'balanced')
    lgr.fit(x_train, y_train)
    pred1 = lgr.predict(x_test)

    pred_list.append(pred1)

    print(accuracy_score(pred1, y_test))
    print(f1_score(pred1, y_test, average = 'macro'))

    svc = LinearSVC(C = 0.0004, class_weight = 'balanced')
    svc.fit(x_train, y_train)
    pred2 = svc.predict(x_test)

    pred_list.append(pred2)

    print(accuracy_score(pred2, y_test))
    print(f1_score(pred2, y_test, average = 'macro'))

    rfc = ensemble.RandomForestClassifier(class_weight = 'balanced')
    rfc.fit(x_train, y_train)
    pred3 = rfc.predict(x_test)

    pred_list.append(pred3)

    print(accuracy_score(pred3, y_test))
    print(f1_score(pred3, y_test, average = 'macro'))

    #gb = ensemble.GradientBoostingClassifier()
    #gb.fit(x_train, y_train)
    #pred = gb.predict(x_test)

    #pred_list.append(pred)

    final_pred = []
    for i in range(len(pred_list[0])):
        temp = [0, 0, 0, 0]
        for j in range(len(pred_list)):
            temp[pred_list[j][i]] += 1

        final_pred.append(np.argmax(temp))
    
    pred = final_pred

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
         
def task1_1_sequence_lab(df):    
    print("======================== task 1_1 =============================")
    
    del df['entity_charOffset']
    del df['entity_id']
    del df['entity_type']
    del df['sentence_id']
    
    print(df.head())
    
    
    df_train, df_test = train_test_split(df, test_size = 0.2, random_state=22, shuffle = False)
    
    text_train = df_train['sentence_text'].as_matrix()
    text_test = df_test['sentence_text'].as_matrix()
    

    
#this is an attempt to concatenate token tags to the dataset, memory leak problems
    #token_name_train = np.column_stack((token_name_train, df_train['token_tag']))
    #token_name_test = np.column_stack((token_name_test, df_test['token_tag']))
    #[:,None]

    GLOVE_DIR = 'C:\\Users\\viktor\\Projects\\Python\\projektHSP\\glove.twitter.27B'
    MAX_SEQUENCE_LENGTH = 30
    MAX_NB_WORDS = 200000
    VALIDATION_SPLIT = 0.2

     
    
    y_train = df_train['entity_name'].as_matrix()
    y_test = df_test['entity_name'].as_matrix()

    y_temp = []
    br = 0
    for i in range(len(text_train)):
        temp = []
        for word in text_train[i].split():
            if MAX_SEQUENCE_LENGTH - len(temp) > 0:
                if word in y_train[i]:
                    br += 1
                    temp.append([0, 1])
                else:
                    temp.append([1, 0])
        if MAX_SEQUENCE_LENGTH - len(temp) > 0:
            for j in range(MAX_SEQUENCE_LENGTH - len(temp)):
                temp.append([1, 0])
        
        y_temp.append(temp)

    y_train = y_temp

    y_temp = []
    y_test2 = []
    br = 0
    for i in range(len(text_test)):
        temp = []
        temp2 = []
        for word in text_test[i].split():
            if MAX_SEQUENCE_LENGTH - len(temp) > 0:
                if word in y_test[i]:
                    br += 1
                    temp.append([0, 1])
                    temp2.append(1)
                    
                else:
                    temp.append([1, 0])
                    temp2.append(0)
        if MAX_SEQUENCE_LENGTH - len(temp) > 0:
            for j in range(MAX_SEQUENCE_LENGTH - len(temp)):
                temp.append([1, 0])
                temp2.append(0)
        
        y_temp.append(temp)
        y_test2.append(temp2)

    y_test = y_temp


    texts = np.reshape(text_train, (-1))
    text_test = np.reshape(text_test, (-1))

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    y_train = np.reshape(y_train, (-1, MAX_SEQUENCE_LENGTH, 2))
    y_test = np.reshape(y_test, (-1, MAX_SEQUENCE_LENGTH, 2))

    x_train = data

    print(x_train.shape)


    sequences_test = tokenizer.texts_to_sequences(text_test)
    x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt'), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    EMBEDDING_DIM = 100

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print(embedding_matrix.shape)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    model = Sequential()

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    model.add(embedding_layer)

    model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, return_sequences = True))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.01)

    print(y_train.shape)

    class_weight = dict()
    class_weight[0] = 0.1
    class_weight[1] = 10


    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'], sample_weight_mode="temporal")

    sample_weight = np.ones((y_train.shape[0], y_train.shape[1]))
    

    for i in range(y_train.shape[0]):
        for j in range(y_train.shape[1]):
            if y_train[i][j][0] == 0: #bitan slucaj
                sample_weight[i,j] = sample_weight[i,j] * 4
            else:
                sample_weight[i,j] = sample_weight[i,j] * 0.5
    

    model.fit(x_train, y_train, nb_epoch=10, batch_size=100, sample_weight=sample_weight)

    print(model.evaluate(x_test, y_test))

    pred = model.predict_classes(x_test)
    y_test2 = np.asarray(y_test2)

    pp_br = 0
    for p in pred:
        for pp in p:
            if pp != 0:
                pp_br += 1

    print(pp_br)

    print(pred.shape)
    print(y_test2.shape)

    print(f1_score(pred, y_test2, average = 'macro'))

    pred = np.reshape(pred, (pred.shape[0] * pred.shape[1], 1))
    y_test2 = np.reshape(y_test2, (pred.shape[0] * pred.shape[1], 1))

    print(f1_score(pred, y_test2, average = 'macro'))
    
if __name__=="__main__":
    df = dfFromXML()
    
    #a = load_from_file(__CHEMSPOT_PATH)
    #x_train, x_test, y_train, y_test, df_train_tag, df_test_tag = task1_1_nltk(df)
    #task1_1_with_tags(x_train, x_test, y_train, y_test, df_train_tag, df_test_tag)
    #task1_1_load_train()
    #task1_2(df)
    task_1_tuttocompleto(df)