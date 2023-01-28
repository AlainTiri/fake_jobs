import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn import svm


class Model_SVM:
    def __init__(self):
        nltk.download('wordnet')
        nltk.download('stopWords')
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = stopwords.words('english')

        # :todo For the V2, dowload the model from the cloud like Cloud Storage
        filename = './app/model/models/model_svm.pkl'
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
            self.perf_of_model = """
                frac=0.07
                re shape
                0    0.611461
                1    0.388539
                Name: fraudulent, dtype: float64
                              precision    recall  f1-score   support
                
                           0       0.77      0.99      0.87       189
                           1       0.99      0.56      0.71       129
                
                    accuracy                           0.82       318
                   macro avg       0.88      0.78      0.79       318
                weighted avg       0.86      0.82      0.80       318
                
                col_0         0   1
                fraudulent         
                0           188   1
                1            57  72"""

        filename_vect = "./app/model//models/vectorizer.pkl"
        with open(filename_vect, 'rb') as f:
            self.vectorizer = pickle.load(f)

    def predict(self, to_predict:str):
        to_predict:list = [self.preprocess(to_predict)]
        vec_to_predict = self.vectorizer.transform(to_predict).toarray()

        return self.model.predict(vec_to_predict)

    def preprocess(self, text):
        if type(text) is float:
            return ""
        text = text.replace("&amp;", "&")
        text = text.replace("!", "! ")
        text = text.replace("\xa0", " ")
        text = text.replace("?", "? ")
        text = text.replace(":", " ")
        text = text.replace("...", " ")
        text = text.replace("  +", " ")
        text = re.sub(r"([a-z\.]{2,})([A-Z])", r"\g<1> \g<2>", text)
        text = re.sub(r"([a-z]{2,})([A-Z])", r"\g<1> \g<2>", text)
        text = text.lower()
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords]

        return ' '.join(tokens)
