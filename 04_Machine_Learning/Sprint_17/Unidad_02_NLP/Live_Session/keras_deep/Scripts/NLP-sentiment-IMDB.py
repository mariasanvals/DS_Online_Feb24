


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import pickle
import numpy as np
print()
# Load dataset
df = pd.read_csv('IMDB Dataset.csv')
#print(df.head(10))
#print(df.info())

# Preprocessing
df['review'] = df['review'].str.lower()

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print('The sentiment data for training is organized in a matrix.')
print('Num. reviews (rows): ',X_train_vec.shape[0])
print('Num. tokens: (columns)',X_train_vec.shape[1])
print()
##################################################################
#
# ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡RELEER y  REESCRIBIR !!!!!!!!!!!!!!!!
# 
#################################################################
X_train_vec_row = X_train_vec.toarray()[1900,:]
print(f'The shape of the container of critics in the processed mathematical form of a matrix is ({X_train_vec.shape[0]},{X_train_vec.shape[1]})')
print(f'The shape of the container of sentiments that correspond to each movie is {y_train.shape}')
print(f'And the type of cells in the critics matrix is {X_train_vec.dtype} (frequencies) and in the sentiments list is {y_train.dtype} (negative/positive).')
print()
print()
print('Each cell, each review-token pair of the matrix is represented with a number, a frequency of appearance of the token (normalized to bla bla bla). Different row-vectors (decomposed reviews in tokens, or words) have different sentiments.That is what the Model is trained to recognize. A row-vector is equivalent to a review. A characteristic of these vectors is that they are sparse: they contain few words, therefore there are many 0s.')
print('Instead, there are: ', len(vectorizer.get_feature_names_out()),' tokens. There are those many tokens.')
print()
print('For example, some of the name of the tokens in this algorithm are: ', vectorizer.get_feature_names_out()[1900:1920])
print()
print(f'If we take a look more closely, for example to review 1900, we can see that it has {len(X_train_vec_row[X_train_vec_row!=0])} cells activated to construct a token list for this particular review. All reviews have the same token names, the same words filled mostly with zeros, and they also have the same number of tokens in the same order. What changes is the content of the row-vector, different reviews (row-vecotrs) have different activation of tokens (cells). We can grasp the amount of the sparsity of the tokens in these critics with the following thought: there is one zero for each word that is not present in the token/word list, in this particular case there are {len(X_train_vec_row[X_train_vec_row!=0])} activated tokens (or non-zero cells) that correspond to the presence of that particular critic (almost a cell for each word in the whole dictionary). But ocassionally, when the word is present in the review, the cell activates for the word that corresponds to the token, the cell then depicts some frequency, and maybe there are other cells with non-zeros, far away. This way the algorithm learns. Each activated token is a lonely number surrounded almost by a full dictionary of zeros, exactly by {len(vectorizer.get_feature_names_out())} of them.')
print(f'For review 1900, the total number of tokens that are not zero is {len(X_train_vec_row[X_train_vec_row!=0])}.')
print('For review 1900, the activated words/tokens are:')
print(vectorizer.get_feature_names_out()[X_train_vec_row!=0])
print(f'The sentiment that corresponds to review 1900 is {y_train.iloc[1900]}.')
print()
print('And that is what we use to teach the Natural Language Processing Model. By itself, it learns to associate different sentiments to reviews, represented or decomposed reviews by lists of words/tokens with some frequency, but usually zero. And we do this for many reviews/sentiments.')
print()
print(f'A curiosity. When we check the original review 1900 we get the following text:\n"{"As much as I dislike saying me too in response to other comments - it's completely true that the first 30 minutes of this film have nothing whatsoever to do with the endless dirge that comprises the following 90. Having been banned somewhere doesn't make a film watchable. Just because it doesn't resemble a Hollywood product does not make it credible. Worse yet, in addition to no discernible plot (other than there are lots of muddy places in Russia and many people, even very old women, drink lots of vodka) a number of visuals are so unnecessarily nauseating I'm in to my second package of Rolaids. As for spoilers - well, the film is so devoid of any narrative thread I couldn't write one if I tried. Don't waste your time or money, and don't confuse this with good Russian cinema."}"')
print('The words do not coincide with the words/tokens in row 1900 of our matrix because of the transformations to which are subjetct the training data.')
print('And the corresponding sentiment is: Negative.')
print('They do not coincide because of the data preparation and shuffling.')
print()
#data_matrix = X_train_vec
# Calcular la media de cada columna
#mean_col = np.mean(data_matrix, axis=0)
# Calcular el rango de cada columna
#range_col = np.ptp(data_matrix, axis=0)
# Calcular la varianza de cada columna (muestra)
#variance_col = np.var(data_matrix, axis=0, ddof=1)
# Calcular la desviación estándar de cada columna (muestra)
#std_dev_col = np.std(data_matrix, axis=0, ddof=1)
#print('np col:', mean_col)
#print('std col:', std_dev_col)
#cv_col = (std_dev_col / mean_col) * 100
#print('cv col:', cv_col)
# Model training
print('Training the model...')
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print()

# Model evaluation
print('Evaluating the model...')
print()
print('The following depicts the ability, the goodness, to tell the correct sentiment of a critic, positive vs negative, for all the test critics, which the Model will now see for the first time. Overall accuracy:')
predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))
print()
#print('The Classification Report')
#print()
print("Classification Report:\n", classification_report(y_test, predictions))
print()
confusion_m = confusion_matrix(y_test,predictions)
print('The Confussion Matrix:')
print('x-axis is predicted while y-axis is true label.')
print()
print(confusion_m)
print()
# Prediction on a new review
#new_review = ["I really enjoyed this movie, it was fantastic!"]
#new_review_2 = ["This movie sucked!"]
#new_review_vec = vectorizer.transform(new_review)
#new_review_vec_2 = vectorizer.transform(new_review_2)

#prediction = model.predict(new_review_vec)
#print("Sentiment prediction for the new review:", prediction)

#prediction_2 = model.predict(new_review_vec_2)
#print("Sentiment prediction for the new review:", prediction_2)
print('Saving the model and the vectorization schema...')
with open('sentiment_model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
print()