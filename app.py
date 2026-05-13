import streamlit as st
import string
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS spam classifier')
input_sms = st.text_area('Enter the message')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = [
        word for word in y
        if word.lower() not in stopwords.words('english')
        and word.lower() not in string.punctuation
    ]

    text = [ps.stem(word) for word in text]

    return " ".join(text)


if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not spam')
