import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle


# Cargar el vectorizador desde el archivo
with open('sentiment_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
# Cargar el vectorizador desde el archivo
with open('vectorizer.pkl', 'rb') as file:
    loaded_vectorizer = pickle.load(file)

# Uso del modelo cargado para predecir el sentimiento de nuevas reseñas
new_review = ["I really enjoyed this movie, it was fantastic!"]
new_review = ["I think this moview is overrated"]
new_review = ["I am not sure what to say about this movie. It did not move me but I enjoyed the ending."]
new_review = ["I am not sure."]


new_review_vec = loaded_vectorizer.transform(new_review)
prediction = loaded_model.predict(new_review_vec)
print("Sentiment prediction for the new review:", prediction)