import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn import svm

lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')

""" Copy past from the notebook """

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
    df.drop_duplicates(inplace=True)
    return df


def preprocess(text):
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
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]

    clean_description = ' '.join(tokens)
    return clean_description


def clean_data(text):
    if type(text) is float:
        return ""
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]

    clean_description = ' '.join(tokens)
    return clean_description


def trainer(df, model, df_fraud, df_true, frac: float):
    print(f"\nfrac = {frac}")
    df_true_sample = df_true.sample(frac=frac, random_state=0)

    df_reshape = pd.concat([df_fraud, df_true_sample])
    # df_reshape = df_reshape.sample(frac=1)
    # check class distribution
    print("re shape")
    print(df_reshape['fraudulent'].value_counts(normalize=True))

    vectorizer = TfidfVectorizer(
        max_features=50000,
        lowercase=False,
        ngram_range=(1, 3))

    X = df_reshape.description
    y = df_reshape.fraudulent

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

    vec = vectorizer.fit(train_X)

    vec_train = vec.transform(train_X)
    vec_train = vec_train.toarray()

    vec_test = vectorizer.transform(test_X).toarray()

    train_data = pd.DataFrame(vec_train, columns=vectorizer.get_feature_names())
    test_data = pd.DataFrame(vec_test, columns=vectorizer.get_feature_names())

    model.fit(train_data, train_y)
    predictions = model.predict(test_data)

    print(classification_report(test_y, predictions))
    # confusion matrix
    print(pd.crosstab(test_y, predictions), end="\n\n")

    return {frac: {"model": model, "vectorizer": vec}}

df = pd.read_csv("dataset.csv", sep = ";")

df['description'] = df['description'].apply(lambda x : clean_data(x))
df = cleandataset(df)

# check class distribution
print(df['fraudulent'].value_counts(normalize = True))

df_fraud = df[df.fraudulent == 1]
df_true = df[df.fraudulent == 0]

frac = [x * 0.01 for x in range(3, 14)]
models = [trainer(df, svm.SVC(), df_fraud, df_true, step) for step in frac]

print("end")

# Modèle sélectionné
# frac = 0.07
# vect avec 50000 features
# SVM non fine-tuned

mymodel = models[4][0.07]
print(mymodel)

text = [preprocess(
    "IC&amp;E Technician | Bakersfield, CA Mt. PosoPrincipal Duties and Responsibilities: Calibrates, tests, maintains, troubleshoots, and installs all power plant instrumentation, control systems and electrical equipment.Performs maintenance on motor control centers, motor operated valves, generators, excitation equipment and motors.Performs preventive, predictive and corrective maintenance on equipment, coordinating work with various team members.Designs and installs new equipment and/or system modifications.Troubleshoots and performs maintenance on DC backup power equipment, process controls, programmable logic controls (PLC), and emission monitoring equipment.Uses maintenance reporting system to record time and material use, problem identified and corrected, and further action required; provides complete history of maintenance on equipment.Schedule, coordinate, work with and monitor contractors on specific tasks, as required.Follows safe working practices at all times.Identifies safety hazards and recommends solutions.Follows environmental compliance work practices.Identifies environmental non-compliance problems and assist in implementing solutions.Assists other team members and works with all departments to support generating station in achieving their performance goals.Trains other team members in the areas of instrumentation, control, and electrical systems.Performs housekeeping assignments, as directed.Conduct equipment and system tagging according to company and plant rules and regulations.Perform equipment safety inspections, as required, and record results as appropriate. Participate in small construction projects.  Read and interpret drawings, sketches, prints, and specifications, as required.Orders parts as needed to affect maintenance and repair.Performs Operations tasks on an as-needed basis and other tasks as assigned.Available within a reasonable response time for emergency call-ins and overtime, plus provide acceptable off-hour contact by phone and company pager.          Excellent Verbal and Written Communications Skills:Ability to coordinate work activities with other team members on technical subjects across job families.Ability to work weekends, holidays, and rotating shifts, as required.")]
vectorizer = mymodel["vectorizer"]
vec_test = vectorizer.transform(text).toarray()
test_data = pd.DataFrame(vec_test, columns=vectorizer.get_feature_names())

print(mymodel["model"].predict(vec_test))
print(

    """
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
    1            57  72""")

# with open("model_svm.pkl", "wb") as f:
#     pickle.dump(mymodel["model"], f)

# with open("vectorizer.pkl", "wb") as f:
#     pickle.dump(mymodel["vectorizer"], f)
