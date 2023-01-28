import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
nltk.download('wordnet')


class Preprocess:
    def __int__(self):
        nltk.download('wordnet')
        nltk.download('stopWords')

    def cleandataset(self, df):
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
        motus = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in motus]

        return ' '.join(tokens)

    def clean_data(self, text):
        if type(text) is float:
            return ""

        self.lemmatizer = WordNetLemmatizer()
        motus = stopwords.words('english')
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in motus]

        return ' '.join(tokens)

    def vectorizer(self, df):
        # BoW-TF:IDF Embedding
        tfidf_vectorizer = TfidfVectorizer(min_df=.02, max_df=.7, ngram_range=[1, 3])

        tpl_tfidf = tfidf_vectorizer.fit_transform(df2['Review_Processed'])
        print(f"Bow-TF:IDF : {tpl_tfidf.shape}")
        df_tfidf = pd.DataFrame(tpl_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names(), index=df2.index)
        df_tfidf.head()


prep = Preprocess()


def init_process() -> pd.DataFrame:
    df = pd.read_csv("dataset.csv", sep = ";")

    df['description'] = df['description'].apply(lambda x : prep.clean_data(x))
    df = prep.cleandataset(df)

    # check class distribution
    print(df['fraudulent'].value_counts(normalize = True))
    return df

# df = init_process()
