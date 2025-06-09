## Libraries
import pandas as pd
import os
import nltk
import numpy as np
import re  
from nltk.stem import wordnet  
from sklearn.feature_extraction.text import CountVectorizer 
from nltk import pos_tag 
from sklearn.metrics import pairwise_distances 
from nltk import word_tokenize
from nltk.corpus import stopwords 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4') 

## Dataset
chat_data = pd.read_excel('C:/Users/Aditya/Desktop/ASHU project/chat/Dataset/dialog_talk_agent.xlsx')
chat_train = pd.read_json('C:/Users/Aditya/Desktop/ASHU project/chat/Dataset/model_data/train.json')

chat_train = chat_train.drop(columns=['viewed_doc_titles', 'used_queries', 'annotations', 'id', 'nq_doc_title'])
chat_train = chat_train.reindex(columns=['question', 'nq_answer'])
chat_train = chat_train.rename(columns={'question': 'Context', 'nq_answer': 'Text Response'})

# removing brackets
def remove_brackets(text):
    new_text = str(text).replace('[', '') 
    new_text = str(new_text).replace(']', '')  
    return new_text

chat_train['Text Response'] = chat_train['Text Response'].apply(remove_brackets) 

# remove apostrophes 
def remove_first_and_last_character(text):
    return str(text)[1:-1]  

chat_train['Text Response'] = chat_train['Text Response'].apply(remove_first_and_last_character)


# data frame 
df = pd.DataFrame() 
column1 = [*chat_data['Context'].tolist(), *chat_train['Context'].tolist()]  
column2 = [*chat_data['Text Response'].tolist(), * chat_train['Text Response'].tolist()]
df.insert(0, 'Context', column1, True)  
df.insert(1, 'Text Response', column2, True)  

# flling missing values
df.ffill(axis = 0, inplace = True)

## Data Preprocessing
def cleaning(x):
    cleaned_array = list()
    for i in x:
        a = str(i).lower() 
        p = re.sub(r'[^a-z0-9]', ' ', a)  
        cleaned_array.append(p)  
    return cleaned_array

df.insert(1, 'Cleaned Context', cleaning(df['Context']), True)

# Data cleaning and lemmatization
def text_normalization(text):
    text = str(text).lower()  
    spl_char_text = re.sub(r'[^a-z]', ' ', text)  
    tokens = nltk.word_tokenize(spl_char_text)  
    lema = wordnet.WordNetLemmatizer()  
    tags_list = pos_tag(tokens, tagset = None) 
    lema_words = []
    for token, pos_token in tags_list:
        if pos_token.startswith('V'): 
            pos_val = 'v'
        elif pos_token.startswith('J'): 
            pos_val = 'a'
        elif pos_token.startswith('R'):  
            pos_val = 'r'
        else:  
            pos_val = 'n'
        lema_token = lema.lemmatize(token, pos_val)  
        lema_words.append(lema_token)  
    return " ".join(lema_words) 

normalized = df['Context'].apply(text_normalization)
df.insert(2, 'Normalized Context', normalized, True)

# removing stopwords
stop = stopwords.words('english')
def removeStopWords(text):
    Q = []
    s = text.split() 
    q = ''
    for w in s:  
        if w in stop:
            continue
        else:  
            Q.append(w)
        q = " ".join(Q)  
    return q

normalized_non_stopwords = df['Normalized Context'].apply(removeStopWords)
df.insert(3, 'Normalized and StopWords Removed', normalized_non_stopwords, True)

## Bag of words
cv = CountVectorizer() 
x_bow = cv.fit_transform(df['Normalized Context']).toarray() 

features_bow = cv.get_feature_names_out()  
df_bow = pd.DataFrame(x_bow, columns = features_bow)  

def chatbot_bow(question):
    tidy_question = text_normalization(removeStopWords(question))  
    cv_ = cv.transform([tidy_question]).toarray()  
    cos = 1- pairwise_distances(df_bow, cv_, metric = 'cosine') 
    index_value = cos.argmax()  
    return df['Text Response'].loc[index_value]