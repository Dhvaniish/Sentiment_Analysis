import pandas as pd
import numpy as np

#importing datasets
dataset = pd.read_csv('C:/Users/dhvan/Downloads/Minor Project/Copy of FitbitData - FitbitData.csv')
dataset.head
dataset.shape

raw_data = dataset[["Reviews","Ratings"]]
raw_data.shape
raw_data.head


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

corpus=[]

for i in range(0, 4000):
  review = re.sub('[^a-zA-Z]', ' ', raw_data['Reviews'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
  
  corpus


clean_text_2 = []

from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

sent_tok = []
for sent in corpus:
    sent = sent_tokenize(sent)
    sent_tok.append(sent)
  
    sent_tok

clean_text_2 = [ word_tokenize(i) for i in corpus ]
clean_text_2

import re
clean_text_3 = []

for words in clean_text_2:
    clean = []
    for w in words :
        res = re.sub(r'[^\w\s]',"", w)
        if res != "":
            clean.append(res)
        clean_text_3.append(clean)
    
clean_text_3

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
clean_text_4 = []
for words in clean_text_3:
    w = []
    for word in words:
            if not word in stopwords.words ('english'):
                w.append (word)
            clean_text_4.append (w)
            
clean_text_4


from nltk.stem.porter import PorterStemmer
port = PorterStemmer()
a = [port.stem(i) for i in ["reading", "washing", "wash", "Driving"]]
a

clean_text_5 = []
for words in clean_text_4:
    w = []
    for word in words:
        w.append(word)
    clean_text_5.append(w)
    
clean_text_5

from nltk.stem.wordnet import WordNetLemmatizer
wnet = WordNetLemmatizer()

import nltk
nltk.download('wordnet')

lem = []
for words in clean_text_4:
    w = []
    for word in words:
        w.append(wnet.lemmatize(word))
    lem.append(w)
    
lem