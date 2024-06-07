import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
@st.cache
def load_data():
    # Load your dataset here (e.g., CSV file)
    df = pd.read_csv("../peliculas.csv")
    return df

# Preprocess your data and train your model
@st.cache(allow_output_mutation=True)
def preprocess_and_train_model(df):
    # Preprocess your data here (e.g., tokenization, vectorization)
    vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
    X = vectorizer.fit_transform(df['review_text'])
    y = df['sentiment']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train your model (e.g., Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, vectorizer, train_accuracy, test_accuracy

# Main function to run the Streamlit app
def main():
    st.title("Movie Review Sentiment Analysis")

    # Load data
    df = load_data()

    # Preprocess data and train model
    model, vectorizer, train_accuracy, test_accuracy = preprocess_and_train_model(df)

    # User input
    review = st.text_area("Enter your movie review:")

    if st.button("Analyze"):
        if review:
            # Vectorize the input text
            review_vectorized = vectorizer.transform([review])

            # Make prediction
            prediction = model.predict(review_vectorized)[0]

            st.write("Prediction:", prediction)
        else:
            st.error("Please enter a movie review.")

if __name__ == "__main__":
    main()