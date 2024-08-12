
# bu kod ilk 30 yorumu analiz ediyor

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd 
import re
import string
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_sentiment(model, tokenizer, max_tokens, text):
    # Metni modele uygun formata dönüştür
    text_tokens = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_tokens, maxlen=max_tokens)
    
    # Tahmin yap
    prediction = model.predict(text_pad)[0][0]
    
    # Tahminin olumlu veya olumsuz olduğunu yorumla
    if prediction >= 0.5:
        sentiment = 'Olumlu 😀'
        probability = prediction * 100
    else:
        sentiment = 'Olumsuz 😞'
        probability = (1 - prediction) * 100
    
    return sentiment, probability

def remove_punctuation(text):
    no_punc = [char for char in text if char not in string.punctuation]
    word_wo_punc = "".join(no_punc)
    return word_wo_punc

def remove_numeric(corpus):
    output = "".join(words for words in corpus if not words.isdigit())
    return output

def main(train_model=True):
    if train_model:
        # Eğitim yap
        df = pd.read_csv("C:/Users/dogab/OneDrive/Desktop/Proje_DuyguAnalizi/turkish_movie_sentiment_dataset.csv")

        # Veriyi işle...

    else:
        # Eğitim yapmadan modeli ve tokenizer'ı yükle
        model = load_model("model.h5")
        with open("tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        
        # max_tokens değerini belirle
        max_tokens = 100

        print("----------------------------------------------------------------------------------------")
        df = pd.read_csv("C:/Users/dogab/OneDrive/Desktop/Proje_DuyguAnalizi/turkish_movie_sentiment_dataset.csv")
        data = df["comment"].values.tolist()

        for i, comment in enumerate(data[:15]):
            predicted_sentiment, probability = predict_sentiment(model, tokenizer, max_tokens, comment)
            print(f"Yorum {i+1}: {comment} - Duygu: {predicted_sentiment}, Olasılık: {probability:.2f}%")

if __name__ == "__main__":
    main(train_model=False) # Eğitim yapmadan tahmin yapmak için False True olarak değiştir!
