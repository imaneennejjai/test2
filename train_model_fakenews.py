import pandas as pd 
import numpy as np 
import pickle
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('news1.csv')
df.columns = ['id','title','text','label']
# print(df.shape)
# print(df.head())
X_train = df['text']
y_train = df['label']
count_vectorizer = CountVectorizer()

count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
test_counts = count_vectorizer.transform(df['text'].values)
test_tfidf = tfidf.transform(test_counts)
X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y_train, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(df['text'],labels,test_size=1)
#tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform train set, transform test set
#tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
#tfidf_test  = tfidf_vectorizer.transform(X_test)
# Initialize a PassiveAggressiveClassifier
pac = MultinomialNB()
pac.fit(X_train,y_train)
# saving vectorizer
with open('count_vectorizer.pickle','wb') as f:
    pickle.dump(count_vectorizer,f)

# saving model
with open('model_fakenews.pickle','wb') as f:
    pickle.dump(pac,f)