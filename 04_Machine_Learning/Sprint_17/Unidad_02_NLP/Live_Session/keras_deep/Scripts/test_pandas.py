


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('IMDB Dataset.csv')
print(df.head(10))
print(df.info())

# Preprocessing
df['review'] = df['review'].str.lower()

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print('Num. documents: ',X_train_vec.shape[0])
print('Num. tokens:',X_train_vec.shape[1])

print('Name of features: ', vectorizer.get_feature_names_out())

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Model evaluation
predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Prediction on a new review
new_review = ["I really enjoyed this movie, it was fantastic!"]
new_review_vec = vectorizer.transform(new_review)
prediction = model.predict(new_review_vec)
print("Sentiment prediction for the new review:", prediction)