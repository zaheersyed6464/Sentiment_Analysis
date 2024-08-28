import pandas as pd

data pd.read_csv('Reviews.csv')

print(data)

#we can review the top 5 rows

data.head()
#Top 10 rows

data.head (10)

data.tail() # Last 5 rows

data.tail(10)

Mata.isnull().sum()
 # checking the null values.

data.duplicated() # checking the duplicated values

combined_text

join(data['Review']) #combine all review text into one string

wordcloud WordCloud (width=800,height 400, background_color= 'white').generate(combined_text)

#Text preprocessing

#coverting a dataset into Lower case

lowercased_text = data['Review'].str.lower()

print(lowercased_text)

#Tokenization is the process of breaking down a pice of text into smaller units, from nltk.tokenize import word_tokenize


From nltk.stem import WordNetLemmatizer

From nltk.corpus import wordnet

Lemmatizer WordNetLemmatizer()

Data['Lemmatized'] = data['Review'].apply(lambda x: '.join([lemmatizer.lemmatize(word,wordnet.v)

Print(data['Lemmatized'])

Normalization

import contractions

data['Expanded'] = data['Review'].apply(contractions.fix)

print(data['Expanded'])

#Removing HTML tags

!pip install beautifulsoup4

from bs4 import BeautifulSoup

data['Cleaned'] = data['Review'].apply(lambda x: BeautifulSoup (x, "html.parser")

from sklearn.feature_extraction.text import TfidfVectorizer vectorizer TfidfVectorizer()

X = vectorizer.fit_transform(data['Review'])

print(X.toarray())

#Building machine Learning Model

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report

vectorizer TfidfVectorizer()

X = vectorizer.fit_transform(data['Review'])

y data 'Liked']

X_train, X_test,y_train,y_test train_test_split(X,y, test_size = 0.2, random_state = 42)

print(X_train, X_test, y_train, y_test)

model MultinomialNB()

model.fit(X_train,y_train)

y_pred model.predict(X_test)

accuracy accuracy_score(y_test,y_pred)

report classification_report(y_test,y_pred)

print(f'Accuracy: {accuracy)')

print('Classification Report:')

print(report)

#predict sentiment for new review

cleaned_review preprocess_text(new_review)

def predict_sentiment(new_review):

X_new = vectorizer.transfrom([cleaned_review])

return model.predict(X_new) [0]

new_reviews= input("Enter a review")

for review in new_reviews:

sentiment predict_sentiment(review)

sentiment_label 'Postive' if sentiment ==1 else 'Negative'

print(f"Review: '[review]'\nPredicted Sentiment: (sentiment_label]\n")
