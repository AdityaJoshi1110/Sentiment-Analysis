
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from flask import Flask, render_template, request

train_data = pd.read_csv(r"C:/Users/adity/PycharmProjects/sentiment analysis/data/train_data.csv")
test_data = pd.read_csv(r"C:/Users/adity/PycharmProjects/sentiment analysis/data/test_data.csv")

def process(review):
    review = BeautifulSoup(review,"html.parser").get_text()
    review = re.sub("[^a-zA-Z]", ' ', review)
    review = review.lower()
    review = review.split()
    swords = set(stopwords.words("english"))
    review = [w for w in review if w not in swords]
    return (" ".join(review))


vectorizer = CountVectorizer(max_features=5000)

X_train = []
for r in range(len(train_data["review"])):
    X_train.append(process(train_data["review"][r]))
X_train = vectorizer.fit_transform(X_train)
X_train = X_train.toarray()

Y_train = np.array(train_data["sentiment"])

X_test = []
for r in range(len(test_data["review"])):
    X_test.append(process(test_data["review"][r]))
X_test = vectorizer.transform(X_test)
X_test = X_test.toarray()

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('details.html')


@app.route('/', methods=['POST'])
def getvalue():
    name = request.form['name']
    movie = request.form['movie']
    review = request.form['review']
    user_review = [ review ]
    X_new = []
    for review in user_review:
        X_new.append(process(review))
    X_new = vectorizer.transform(X_new)
    X_new = X_new.toarray()
    prediction = model.predict(X_new)
    for analysis in prediction:
        sentiment = analysis

    sentiment = np.where(sentiment, 'Positive', 'Negative')
    return render_template('sentiment.html', name=name, movie=movie, review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
