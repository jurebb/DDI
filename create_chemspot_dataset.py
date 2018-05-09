import molminer
from molminer import ChemSpot
import xml.etree.ElementTree as ET 
import os
import pandas as pd
import numpy as np
from xlm_parsers_functions import *

__PARENT_PATH = '../Train/DrugBank/'


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

def write_to_file():
    with open('Exported.txt', 'w') as file:
        for ent in range(len(entities_dataset)):
            file.write(entities_dataset[ent][1] + '\n')
         
def ChemSpot_RUN():
    chemspot = ChemSpot()
    #print(chemspot.help())  # this will show the output of "$ chemspot -h"
    #print(chemspot._OPTIONS_REAL)  # this will show the mapping between ChemSpot class and real ChemSpot parameters

    
    entity = "Before using this medication, tell your doctor or pharmacist of all prescription and nonprescription products you may use, especially of: aminoglycosides (e.g., gentamicin, amikacin), amphotericin B, cyclosporine, non-steroidal anti-inflammatory drugs (e.g., ibuprofen), tacrolimus, vancomycin."
    #entity2 = "Aminosalicylic acid may decrease the amount of digoxin (Lanoxin, Lanoxicaps) that gets absorbed into your body."
    #entity3 = "Uricosuric Agents: Aspirin may decrease the effects of probenecid, sulfinpyrazone, and phenylbutazone."
    #entity4 = "Pyrazolone Derivatives (phenylbutazone, oxyphenbutazone, and possibly dipyrone): Concomitant administration with aspirin may increase the risk of gastrointestinal ulceration."
    entity5 = "use potassium supplements if necessary."
    chemspot_values = chemspot.process(input_text=entity5)
        
    return chemspot_values

def Extractor_RUN():
    extractor = molminer.Extractor()
    print(extractor.process(input_file='Exported.txt'))
    
def task1_1_chemspot(df):    
    chemspot = ChemSpot()
    
    del df['entity_charOffset']
    del df['entity_id']
    del df['entity_name']
    del df['entity_type']
    
    df_train, df_test = train_test_split(df.drop_duplicates().as_matrix(), test_size = 0.2, random_state=22)
    
    text_train = df_train
    text_test = df_test
    
    #print(np.unique(df['sentence_text'].as_matrix()).shape)
    print(text_train.shape)
    print(text_test.shape)
    
    print(len(str(text_test[12][1]).split()) == 1)
    print(text_test[13])
    #'-1|-1|\N|\N'
    counter = 1
    filename='task1_1_chemspot_test.txt'
    with open(filename, 'w') as file:
        print('writing to file:', filename)
        for sentence in text_test:
            #print(sentence)
            try:
                chemspot_values = chemspot.process(input_text=sentence[1])
            except:
                print('failed sentence', sentence[0])
                counter += 1
                continue
            
            file.write(str('#sid# ' + sentence[0]))
            file.write('\n')
            file.write(str('#s# ' + sentence[1]))
            for i in chemspot_values['content']:
                file.write('\n')
                file.write(str(i['start'] + '|' + i['end'] + '|' + i['entity'] + '|' + i['type']))
            counter += 1
            file.write('\n')
            if(counter % 5 == 0):
                print('chemspot: finished sentence', counter, 'of', text_test.shape[0])
                
                
    counter = 1
    filename='task1_1_chemspot_train.txt'
    with open(filename, 'w') as file:
        print('writing to file:', filename)
        for sentence in text_train:
            #print(sentence)
            try:
                chemspot_values = chemspot.process(input_text=sentence[1])
            except:
                print('failed sentence', sentence[0])
                counter += 1
                continue
            
            file.write(str('#sid# ' + sentence[0]))
            file.write('\n')
            file.write(str('#s# ' + sentence[1]))
            for i in chemspot_values['content']:
                file.write('\n')
                file.write(str(i['start'] + '|' + i['end'] + '|' + i['entity'] + '|' + i['type']))
            counter += 1
            file.write('\n')
            if(counter % 5 == 0):
                print('chemspot: finished sentence', counter, 'of', text_train.shape[0])
                
        
if __name__=="__main__":
    df = dfFromXML()
    
    task1_1_chemspot(df)
    
