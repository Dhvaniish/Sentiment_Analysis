import numpy as np
import bz2

def get_labels_and_texts(file):
    labels = []
    texts = []
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[0]) - 1)
        texts.append(x[10:].strip())
    return np.array(labels), texts

train_labels, train_texts = get_labels_and_texts('Copy of FitbitData - FitbitData.tar.bz2')
test_labels, test_texts = get_labels_and_texts('test.ft.txt.bz2')

train_labels[0]
train_texts[0]

train_labels=train_labels[0:4000]
train_texts=train_texts[0:4000]

import re
NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts
        
train_texts = normalize_texts(train_texts)
test_texts = normalize_texts(test_texts)

train_texts[0]

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(train_texts)
X = cv.transform(train_texts)
X_test = cv.transform(test_texts)

X_test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(
    X, train_labels, train_size = 0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))
    
lr.predict(X_test[29])

test_labels[29]

test_texts[29]