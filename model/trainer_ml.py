from sklearn import svm

import pandas as pd

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump


def cleandataset(df):
    df["description"] = df["description"].str.replace("&amp;", "&", regex=False)
    df["description"] = df["description"].str.replace("\xa0", " ", regex=False)
    df["description"] = df['description'].str.replace("!", "! ", regex=False)
    df["description"] = df['description'].str.replace("?", "? ", regex=False)
    df["description"] = df['description'].str.replace(":", " ", regex=False)
    df["description"] = df['description'].str.replace("...", " ", regex=False)
    df["description"] = df['description'].str.replace("  +", " ", regex=True)
    df["description"] = df['description'].str.replace("([a-z]{2,})([A-Z])", "\g<1> \g<2>", regex=True)
    df["description"] = df['description'].str.replace("([a-z\.]{2,})([A-Z])", "\g<1> \g<2>", regex=True)
    df["description"] = df['description'].str.lower()

    df.dropna(inplace=True)
    #     df.drop_duplicates(inplace=True)
    return df


def clean_data(text):
    #     text = re.sub('[^a-zA-Z]' , ' ' , text)
    if type(text) is float:
        return ""
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if not word in stopwords]
    clean_description = ' '.join(tokens)
    return clean_description


lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')
# nltk.download('wordnet')

df = pd.read_csv("dataset.csv", sep=";")


df['description'] = df['description'].apply(lambda x: clean_data(x))

# print(df.shape)
df = cleandataset(df)
# print(df.shape)

# check class distribution
# print(df['fraudulent'].value_counts(normalize=True))

df_fraud = df[df.fraudulent == 1]
df_true = df[df.fraudulent == 0]

# df_fraud.shape

df_true_sample = df_true.sample(frac=0.05, random_state=0)
# print(df_true.shape)
# print(df_true_sample.shape)

df_reshape = pd.concat([df_fraud, df_true_sample])
df_reshape = df_reshape.sample(frac=1)
# check class distribution
df_reshape['fraudulent'].value_counts(normalize=True)

vectorizer = TfidfVectorizer(
    max_features=50000,
    lowercase=False,
    ngram_range=(1, 3))

X = df_reshape.description
y = df_reshape.fraudulent

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

vec_train = vectorizer.fit_transform(train_X)
vec_train = vec_train.toarray()

vec_test = vectorizer.transform(test_X).toarray()

train_data = pd.DataFrame(vec_train, columns=vectorizer.get_feature_names())
test_data = pd.DataFrame(vec_test, columns=vectorizer.get_feature_names())

""" Training model SVM """
model_svm = svm.SVC()

model_svm.fit(train_data, train_y)
predictions = model_svm.predict(test_data)

print(classification_report(test_y , predictions))
# confusion matrix
print(pd.crosstab(test_y, predictions))

dump(model_svm, 'model_svm.joblib')
