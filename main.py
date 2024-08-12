import numpy as np
import re
import nltk
import pandas as pd
import nltk as nlp
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

stopWords = set(stopwords.words('turkish'))

def pre_processing(text):
    text = text.lower()  
    text = re.sub("[^abcçdefgğhıijklmnoöprsştuüvyz]"," ",text)
    text = nltk.word_tokenize(text) 
    text = [word for word in text if not word in set(stopWords)] 
    lemma = nlp.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    return text

df_train = pd.read_csv('train.csv', encoding='unicode_escape')
df_test = pd.read_csv('test.csv', encoding='unicode_escape')

df_train["clean_text"] = df_train["comment"].apply(lambda x: pre_processing(x))
df_test["clean_text"] = df_test["comment"].apply(lambda x: pre_processing(x))

X_train = df_train["clean_text"]
X_test = df_test["clean_text"]
y_train = df_train["Label"]
y_test = df_test["Label"]

#print("x_train", X_train.shape)
#print("x_test", X_test.shape)
#print("y_train", y_train.shape)
#print("y_test", y_test.shape)

LogisticRegressionModel = Pipeline([('tfidf', TfidfVectorizer()),('clf', LogisticRegression())])
LogisticRegressionModel.fit(X_train, y_train)

def plot_confusion_matrix(Y_test, Y_preds):

    conf_mat = confusion_matrix(Y_test, Y_preds)
    fig = plt.figure(figsize=(6,6))
    plt.matshow(conf_mat, cmap=plt.cm.Blues, fignum=1)
    plt.yticks(range(2), range(2))
    plt.xticks(range(2), range(2))
    plt.colorbar()
    
    for i in range(2):
        for j in range(2):
            plt.text(i-0.2,j+0.1, str(conf_mat[j, i]), color='tab:red')

cv_scores = cross_val_score(LogisticRegressionModel, X_train, y_train, cv=10)
#print("CV average score: %.2f" % cv_scores.mean())

result = LogisticRegressionModel.predict(X_test)
cr = classification_report(y_test, result)
#print(cr) 

#print('Train Accuracy : %.3f' % LogisticRegressionModel.score(X_train, y_train))
#print('Test Accuracy : %.3f' % LogisticRegressionModel.score(X_test, y_test))

y_pred = LogisticRegressionModel.predict(X_test)
#print(precision_score(y_test, y_pred, average='macro'), ": is the precision score")
#print(recall_score(y_test, y_pred, average='macro'), ": is the recall score")
#print(f1_score(y_test, y_pred, average='macro'), ": is the f1 score")

plot_confusion_matrix(y_test, LogisticRegressionModel.predict(X_test))

input_text = input("Lütfen analiz etmek istediğiniz metni girin: ")

clean_input_text = pre_processing(input_text)

prediction = LogisticRegressionModel.predict([clean_input_text])
proportion = LogisticRegressionModel.predict_proba([clean_input_text])

if prediction[0] == 1:
    print("Girdi metni pozitif duygulu.")
    print("Tahmin olasılığı:", proportion[0][1])
else:
    print("Girdi metni negatif duygulu.")
    print("Tahmin olasılığı:", proportion[0][0])
