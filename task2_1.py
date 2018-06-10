import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble

from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import Adam

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

import nltk 
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET 
import os
from xlm_parsers_functions import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors
import math

def my_lstm(x_train, x_test, y_train):

    x_train_str = []
    for podatak in x_train:
        x_train_str.append(podatak[0] + podatak[1] + podatak[2])
    x_train_str = np.asarray(x_train_str)

    x_test_str = []
    for podatak in x_test:
        x_test_str.append(podatak[0] + podatak[1] + podatak[2])
    x_test_str = np.asarray(x_test_str)


    GLOVE_DIR = 'C:\\Users\\viktor\\Projects\\Python\\projektHSP\\glove.twitter.27B'
    MAX_SEQUENCE_LENGTH = 30
    MAX_NB_WORDS = 200000
    VALIDATION_SPLIT = 0.2

    texts = x_train_str

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = y_train

    x_train = data
    y_train = labels


    sequences_test = tokenizer.texts_to_sequences(x_test_str)
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

    model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.01)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print(y_train.shape)

    model.fit(x_train, y_train, nb_epoch=30, batch_size=100)

    return model.predict_classes(x_test, 100)


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

    #print(df2.head())
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



    data = []

    for i in range(len(data1)):
        if len(data1[i]) > 1:
            for j in range(len(data1[i])):
                for k in range(j + 1, len(data1[i])):
                    data.append([data1[i][j][0], data1[i][j][1], data1[i][k][1], 'None']) #dodajemo tuple (recenica, entitet_j, entitet_k, 'None')


    for i in range(len(data)):
        for dd in df.as_matrix():
            d = data[i]
            if (dd[2] == d[2] and dd[1] == d[1]) or (dd[2] == d[1] and dd[1] == d[2]):
                data[i][3] = dd[3]
                break


    return data


def simple_nn(x_train, x_test, y_train):

    print(x_train.shape)

    model = Sequential()

    model.add(Dense(100, input_dim = x_train.shape[1]))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    adam = Adam(lr = 0.01)

    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['accuracy']) 

    model.fit(x_train, y_train, nb_epoch = 20, batch_size = 100)

    pred = model.predict_classes(x_test, 100) #pred_temp je formata [[1], [0.5], [-0.7], ...] umjesto normalnog niza predikcija [1, 0.5, -0.7, ...] 

    return pred


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

    print(df.head(200))

    df['interection_type'][df['interection_type'] != 'None'] = 'interaction'

    print('novo')
    print(df.head(200))
    
    df_train, df_test = train_test_split(df, test_size = 0.2, shuffle = False)

    #print(df_train.head())
    print(df_train.shape)

    x_train = df_train['sentence_text'].as_matrix().reshape(-1, 1)
    x_test = df_test['sentence_text'].as_matrix().reshape(-1, 1)

    entity1_name_train = df_train['entity1_name'].as_matrix().reshape(-1, 1)
    entity1_name_test = df_test['entity1_name'].as_matrix().reshape(-1, 1)
    entity2_name_train = df_train['entity2_name'].as_matrix().reshape(-1, 1)
    entity2_name_test = df_test['entity2_name'].as_matrix().reshape(-1, 1)

    x_train = np.concatenate((x_train, entity1_name_train), axis=1)
    x_test = np.concatenate((x_test, entity1_name_test), axis=1)

    x_train = np.concatenate((x_train, entity2_name_train), axis=1)
    x_test = np.concatenate((x_test, entity2_name_test), axis=1)

    y_train = df_train['interection_type'].astype("category").cat.codes.as_matrix()
    y_test = df_test['interection_type'].astype("category").cat.codes.as_matrix()

    lb = OneHotEncoder()

    y_train = lb.fit_transform(y_train.reshape(-1, 1))
    y_test = lb.transform(y_test.reshape(-1, 1))


    pred = my_lstm(x_train, x_test, y_train)

    print(pred)
    print(y_test)

    #pred = lb.inverse_transform(pred)
    y_train = df_train['interection_type'].astype("category").cat.codes.as_matrix()
    y_test = df_test['interection_type'].astype("category").cat.codes.as_matrix()
 

    pred_list = []
    #pred_list.append(pred)

    print(pred)
    print(y_test)

    print(accuracy_score(pred, y_test))
    print(f1_score(pred, y_test, average = 'macro'))


    x_train = df_train['sentence_text'].as_matrix()
    x_test = df_test['sentence_text'].as_matrix()

    entity1_name_train = df_train['entity1_name'].as_matrix()
    entity1_name_test = df_test['entity1_name'].as_matrix()
    entity2_name_train = df_train['entity2_name'].as_matrix()
    entity2_name_test = df_test['entity2_name'].as_matrix()


    sw = stopwords.words("english")
    vectorizer = TfidfVectorizer(lowercase=True, stop_words = sw, binary = True, sublinear_tf  = True, norm = None)

    x_train = vectorizer.fit_transform(x_train).toarray()
    x_test = vectorizer.transform(x_test).toarray()

    print(x_train.shape)

    entity1_name_train = vectorizer.transform(entity1_name_train).toarray()
    entity1_name_test = vectorizer.transform(entity1_name_test).toarray()

    entity2_name_train = vectorizer.transform(entity2_name_train).toarray()
    entity2_name_test = vectorizer.transform(entity2_name_test).toarray()


    x_train = np.concatenate((x_train, entity1_name_train), axis=1)
    x_test = np.concatenate((x_test, entity1_name_test), axis=1)

    x_train = np.concatenate((x_train, entity2_name_train), axis=1)
    x_test = np.concatenate((x_test, entity2_name_test), axis=1)

    y_train = df_train['interection_type'].astype("category").cat.codes.as_matrix()
    y_test = df_test['interection_type'].astype("category").cat.codes.as_matrix()

    y_train = lb.transform(y_train.reshape(-1, 1))
    y_test = lb.transform(y_test.reshape(-1, 1))

    pred1 = simple_nn(x_train, x_test, y_train)

    #pred = lb.inverse_transform(pred)
    y_train = df_train['interection_type'].astype("category").cat.codes.as_matrix()
    y_test = df_test['interection_type'].astype("category").cat.codes.as_matrix()
 
    pred_list.append(pred1)

    print(accuracy_score(pred1, y_test))
    print(f1_score(pred1, y_test, average = 'macro'))

    lgr = LogisticRegression(C = 0.05, class_weight = 'balanced')
    lgr.fit(x_train, y_train)
    pred2 = lgr.predict(x_test)

    pred_list.append(pred2)

    print(accuracy_score(pred2, y_test))
    print(f1_score(pred2, y_test, average = 'macro'))

    svc = LinearSVC(C = 0.0004, class_weight = 'balanced')
    svc.fit(x_train, y_train)
    pred3 = svc.predict(x_test)

    pred_list.append(pred3)

    print(accuracy_score(pred3, y_test))
    print(f1_score(pred3, y_test, average = 'macro'))

    rfc = ensemble.RandomForestClassifier(n_estimators = 30, min_samples_split=6, max_features=0.1, class_weight = 'balanced')
    rfc.fit(x_train, y_train)
    pred4 = rfc.predict(x_test)

    pred_list.append(pred4)

    print(accuracy_score(pred4, y_test))
    print(f1_score(pred4, y_test, average = 'macro'))

    #gb = ensemble.GradientBoostingClassifier()
    #gb.fit(x_train, y_train)
    #pred = gb.predict(x_test)

    #pred_list.append(pred)

    final_pred = []
    for i in range(len(pred_list[0])):
        temp = [0, 0]
        for j in range(len(pred_list)):
            temp[pred_list[j][i]] += 1

        final_pred.append(np.argmax(temp))

    print(accuracy_score(final_pred, y_test))
    print(f1_score(final_pred, y_test, average = 'macro'))
            
            


main()