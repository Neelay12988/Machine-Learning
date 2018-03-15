# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
# This imports the stop words to be eliminated
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    # Keep only alpha character in datasets because special characters like ! will be of use also pizza! and pizza would create different columns but will be of no use
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    
    # Removing upper cases as Pizza and pizza are the same and shouldn't create different columns in corpus
    review = review.lower()
    
    # Converting a line to list of individual words for steaming
    review = review.split()
    
    # Words like 'that', 'this' etc used in natural language is of no use for analysis hence removing them
    # Reducing the corpus with steaming becasue 'liked' and 'like' which convey the same meaning shouldn't be treated as 2 seperate words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    # This is raw review without unneccessary words
    review = ' '.join(review)
    
    # Buildting corpus to create bag of words
    corpus.append(review)

# Creating the Bag of Words model
# Bag of words is a model that consists of every unique words from the text as individual column and the cell is populated with counter of word usage in that particular review
# Max features is max number of columns in the bag of words. This will reduce Sparcity by removing words with lower frequencies, hence words that might have occured in the reviews just once or twice
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)

# X is independent variable and consists of all the unique necessary words used in the reviews
X = cv.fit_transform(corpus).toarray()
# y is the dependent 
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
# Using model_selection instead of cross_validation because cross_validation is decomisioned in latest version of sklearn library
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Naive Bayes Classification Model to the Training set but can use any classification model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# For Decision Tree Classification comment other classifiers and use as follows:
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

# For Random Forest Classification comment other Classifiers and use as follows:
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
