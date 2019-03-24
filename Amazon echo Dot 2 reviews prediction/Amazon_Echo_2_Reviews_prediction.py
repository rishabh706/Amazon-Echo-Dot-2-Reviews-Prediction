# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:09:24 2018

@author: rishabh rahatgaonkar
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv("Amazon Echo 2 Reviews.csv")
dataset=dataset.loc[:,('Title','Review Text','Rating')]

# Cleaning the text

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def get_rating_category(rating):
	if rating >=3:
		return 1
	
	else:
		return 0
		


for i in range(0,len(dataset)):
    dataset.loc[i,'cleaned_title']=re.sub('[^a-zA-Z]',' ',str(dataset['Title'][i]))
    dataset.loc[i,'cleaned_text']=re.sub('[^a-zA-Z]',' ',str(dataset['Review Text'][i]))
    dataset.loc[i,'cleaned_title']=dataset.loc[i,'cleaned_title'].lower()
    dataset.loc[i,'cleaned_text']=dataset.loc[i,'cleaned_text'].lower()
    dataset.loc[i,'cleaned_title'].split()
    dataset.loc[i,'cleaned_text'].split()
    ps=PorterStemmer()
    cleaned_title=dataset.loc[i,'cleaned_title']
    cleaned_text=dataset.loc[i,'cleaned_text']
    cleaned_title=[ps.stem(word) for word in cleaned_text if not word in set(stopwords.words('english'))]
    cleaned_title=' '.join(cleaned_title)
    cleaned_text=[ps.stem(word) for word in cleaned_text if not word in set(stopwords.words('english'))]
    cleaned_text=' '.join(cleaned_text)


dataset['rating_category']=dataset['Rating'].apply(lambda x: get_rating_category(x))

# Creating the bag of words model

from sklearn.feature_extraction.text import CountVectorizer
bow_vector=CountVectorizer(max_features=1500)
bow_matrix=bow_vector.fit_transform(dataset['cleaned_text'])
print(bow_matrix)

bow_vector_features = bow_vector.get_feature_names()
bow_matrix = bow_matrix.toarray()
bow_df = pd.DataFrame(bow_matrix, columns = bow_vector_features)

#X=dataset.loc[:,['cleaned_title','cleaned_text']]
X=bow_df.copy()
y=dataset.loc[:,['rating_category']]
#y=y['rating_category'].astype(object)
# Splitting the dataset into the training set and the test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
'''
'''
# Fitting the Decision Tree classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
'''

# Fitting the Naive_bayes classifier into the training set
'''
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train.values.ravel())
'''
'''
# Fitting the SVM classifier
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train.values.ravel())
'''

# Fitting the Logistic Regression into the Training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(class_weight='balanced')
classifier.fit(X_train,y_train)
	 
# Predicting the test set results
y_pred=classifier.predict(X_test)
#THRESHOLD=2.22e-15
#preds=np.where(classifier.predict(X_test) > THRESHOLD)
#y_pred_proba=classifier.predict_proba(X_test)
#y_pred_proba=y_pred_proba[:,0]
   
     
# Making the confusion matrix 
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,precision_recall_curve
y_pred=pd.DataFrame(y_pred)
#preds=pd.DataFrame(list(preds))
cm=confusion_matrix(y_test,y_test)
print(accuracy_score(y_test,y_test))
print(classification_report(y_test,y_pred))
#cm=confusion_matrix(y_test,list(preds))
#print(accuracy_score(y_test,preds))
#print(classification_report(y_test,preds)

#precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)




















