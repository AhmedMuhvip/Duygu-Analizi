import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd 
import re
import string
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_sentiment(model, tokenizer, max_tokens, text):

    text_tokens = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_tokens, maxlen=max_tokens)
    
    prediction = model.predict(text_pad)[0][0]

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
        df = pd.read_csv("C:/Users/dogab/OneDrive/Desktop/Proje_DuyguAnalizi/turkish_movie_sentiment_dataset.csv")

        comments = lambda x : x[23:-24]
        df["comment"] = df["comment"].apply(comments)

        floatize = lambda x : float(x[0:-2])
        df["point"] = df["point"].apply(floatize)
        df.drop(df[df["point"] == 3].index, inplace = True)
        df["point"] = df["point"].replace(1, 0)
        df["point"] = df["point"].replace(2, 0)
        df["point"] = df["point"].replace(4, 1)
        df["point"] = df["point"].replace(5, 1)

        df.reset_index(inplace = True)
        df.drop("index", axis = 1, inplace = True)

        df["comment"] = df["comment"].apply(lambda x: x.lower())
        df["comment"] = df["comment"].apply(lambda x: remove_punctuation(x))
        df["comment"] = df["comment"].apply(lambda x: x.replace("\r", " "))
        df["comment"] = df["comment"].apply(lambda x: x.replace("\n", " "))
        df["comment"] = df["comment"].apply(lambda x: remove_numeric(x)) 

        target = df["point"].values.tolist()
        data = df["comment"].values.tolist()

        cutoff = int(len(data)*0.80)

        X_train, X_test = data[:cutoff], data[cutoff:]
        y_train, y_test = target[:cutoff], target[cutoff:]

        num_words = 10000
        tokenizer = Tokenizer(num_words = num_words)
        tokenizer.fit_on_texts(data)

        X_train_tokens = tokenizer.texts_to_sequences(X_train)
        X_test_tokens = tokenizer.texts_to_sequences(X_test)

        num_tokens = [len(tokens) for tokens in X_train_tokens + X_test_tokens]
        num_tokens = np.array(num_tokens)

        max_tokens = int(np.mean(num_tokens) + (2*np.std(num_tokens)))

        X_train_pad = pad_sequences(X_train_tokens, maxlen=max_tokens) 
        X_test_pad = pad_sequences(X_test_tokens, maxlen=max_tokens)

        idx = tokenizer.word_index
        inverse_map = dict(zip(idx.values(), idx.keys()))

        embedding_size = 50
        model = Sequential()
        model.add(Embedding(input_dim=num_words, output_dim=embedding_size, input_length=max_tokens, name="embedding_layer"))
        model.add(GRU(units=16, return_sequences=True))
        model.add(GRU(units=8, return_sequences=True))
        model.add(GRU(units=4))
        model.add(Dense(1, activation="sigmoid"))

        optimizer = Adam(learning_rate=1e-3)
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        X_train_pad = np.array(X_train_pad)
        y_train = np.array(y_train)

        model.fit(X_train_pad, y_train, epochs=5, batch_size=256)

        model.save("model.h5")
        with open("tokenizer.pickle", "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        model = load_model("model.h5")
        with open("tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        
        max_tokens = 100

        print("--------------------------------------------------------")
        user_input = input("Lütfen bir metin girin: ")
        
        predicted_sentiment, probability = predict_sentiment(model, tokenizer, max_tokens, user_input)
        print(f"Girilen metnin duygusu: {predicted_sentiment} -> Olasılık: % {probability:.2f}\n")

if __name__ == "__main__":
    main(train_model=False) # Eğitim yapmadan tahmin yapmak için False True olarak değiştir!
