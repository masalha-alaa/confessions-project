import pandas as pd
from output.helpers import *
from datetime import datetime
import emoji
import re
import string
import nltk
from nltk import ngrams, FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn import svm
from sklearn import naive_bayes as nb
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


class Columns:
    ID = 'id'
    TIME = 'time'
    TEXT = 'text'
    LIKES = 'likes'
    REACTIONS = 'reactions'
    _TS_ADD = '_ts'
    YEAR = 'year' + _TS_ADD
    MONTH = 'month' + _TS_ADD
    DAY = 'weekday' + _TS_ADD
    HOUR = 'hour' + _TS_ADD
    POST_LEN = 'post_length'
    DISTINCT = 'distinct_words'
    AVG_WRD = 'avg_wrd_len'
    MAX_WRD = 'max_wrd_len'
    MIN_WRD = 'min_wrd_len'
    NGRAMS = 'ngrams'
    NUMBERS = 'numbers'
    EX_MARKS = 'exclamation'


def clean(text):
    def remove_emojis(txt):
        return emoji.get_emoji_regexp().sub(r'EMJ', txt)

    def remove_punctuation(txt):
        return txt.translate(str.maketrans('', '', string.punctuation))

    def remove_url(txt):
        return re.sub(r'\(?http\S+\)?', 'URL', remove_emojis(txt))

    def squeeze_spaces(txt):
        return re.sub(r'\s{2,}', ' ', txt)

    def trim(txt):
        return txt.strip()

    def remove_quotes(txt):
        # return re.sub(r'[\"\']', '', txt)
        return re.sub(r'\"', '', txt)

    text = remove_emojis(text)
    text = remove_url(text)
    # text = remove_punctuation(text)
    text = remove_quotes(text)
    text = squeeze_spaces(text)
    text = trim(text)

    return text


def k_most_common_ngrams(X, ng=2, k=20):
    """
    Return most K common trigrams in the files in the input directory.
    """
    top_k = FreqDist(ngrams('\n'.join(X).split(), ng))
    return [' '.join(t[0]) for t in top_k.most_common(k)]


def k_most_common_char_ngrams(X, ng=2, k=20):
    """
    Return most K common trigrams in the files in the input directory.
    """
    top_k = FreqDist(ngrams('\n'.join(X), ng))
    return [' '.join(t[0]) for t in top_k.most_common(k)]


def get_most_common_k(X, k=50):
    counter = Counter(" ".join(X.apply(lambda s: re.sub(f'[{string.punctuation}]', '', s).strip())).split())
    return sorted(counter, key=counter.get, reverse=True)[:k]


def get_best_k(X, y, k=20):
    # Crete TF-IDF values based on the training data
    vectorizer = TfidfVectorizer(use_idf=True)
    # vectorizer = CountVectorizer()
    tfidf_vals = vectorizer.fit_transform(X)

    # create and fit selector
    selector = SelectKBest(k=k)
    selector.fit(tfidf_vals.toarray(), y)
    words_idx = selector.get_support(indices=True)

    # get the actual words
    the_words = [k for k,v in vectorizer.vocabulary_.items() if v in words_idx]

    return the_words


def my_train_test_split(df, label_col_name, test_percentage, shuffle=False):
    df2 = df.copy()
    if shuffle:
        df2 = df2.sample(frac=1).reset_index(drop=True)

    data_cols = [c for c, col in enumerate(df2.columns) if col != label_col_name]
    train_size = int(df2.shape[0] * (1 - test_percentage))

    X_train = df2.iloc[:train_size, data_cols]
    y_train = df2.iloc[:train_size][label_col_name]
    X_test = df2.iloc[train_size:, data_cols].reset_index(drop=True)
    y_test = df2.iloc[train_size:][label_col_name].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def classify(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    if hasattr(clf, 'feature_importances_'):
        forest_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    else:
        forest_importances = None
    return train_score, test_score, forest_importances


def run_classifiers(X_train, y_train, X_test, y_test):
    # calculate baseline
    print('Baseline (random chance):')
    random_chance = y_train.value_counts().max() / len(y_train)
    print(f'Train dataset: {random_chance*100:.2f}%')
    random_chance = y_test.value_counts().max() / len(y_test)
    print(f'Test dataset: {random_chance*100:.2f}%')
    print('')

    SVM = False
    NB = False
    DT = False
    RF = True
    KNN = False

    # SVM
    if SVM:
        print('SVM')
        train_score, test_score, _ = classify(svm.SVC(), X_train, y_train, X_test, y_test)
        print(f'Training accuracy: {train_score}')
        print(f'Test accuracy: {test_score}')

    # Naive Bayes
    if NB:
        print('Naive Bayes')
        train_score, test_score, _ = classify(nb.MultinomialNB(), X_train, y_train, X_test, y_test)
        print(f'Training accuracy: {train_score}')
        print(f'Test accuracy: {test_score}')

    # Decision Tree
    if DT:
        print('Decision Tree')
        train_score, test_score, ftr_impr = classify(tree.DecisionTreeClassifier(), X_train, y_train, X_test, y_test)
        print(f'Training accuracy: {train_score}')
        print(f'Test accuracy: {test_score}')

    # Random Forest
    if RF:
        print('Random Forest')
        # max_depth=10, min_samples_leaf=2 reduce overfitting very well (from 99% on training to 70%, without harming test)
        train_score, test_score, ftr_impr = classify(RandomForestClassifier(max_depth=10, min_samples_leaf=2, random_state=SEED),
                                           X_train, y_train, X_test, y_test)
        print(f'Training accuracy: {train_score}')
        print(f'Test accuracy: {test_score}')
        print(f'Feature importances:\n{ftr_impr.sort_values(ascending=False)[:20]}')

    # KNN
    if KNN:
        print('KNN')
        train_score, test_score, _ = classify(neighbors.KNeighborsClassifier(), X_train, y_train, X_test, y_test)
        print(f'Training accuracy: {train_score}')
        print(f'Test accuracy: {test_score}')


if __name__ == '__main__':
    READ_PREPROCESSED = True
    SAVE_TOKENIZED = True  # only if READ_TOKENIZED is False
    SEED = 42

    input_path = Helpers.get_csv_path()

    if READ_PREPROCESSED:
        input_path = str(input_path).replace('.csv', ' preprocessed.csv')
        print('Reading preprocessed csv...')

    df = pd.read_csv(input_path)

    if not READ_PREPROCESSED:
        print('Preprocessing...')
        ts = datetime.now()

        # clean text
        df[Columns.TEXT] = df[Columns.TEXT].apply(clean)

        # tokenize
        df[Columns.TEXT] = df[Columns.TEXT].apply(lambda s: ' '.join(nltk.word_tokenize(s.lower())))

        if SAVE_TOKENIZED:
            df.to_csv(str(input_path).replace('.csv', ' preprocessed.csv'), index=False)
            print('Preprocessed saved.')
            print(f'Time: {datetime.now() - ts}')

    # create target
    TARGET = LABEL = 'popular'
    df[TARGET] = df[Columns.LIKES] >= 9

    # drop unused columns
    df.drop([Columns.REACTIONS, Columns.LIKES], axis=1, inplace=True)

    # shuffle
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print('Adding features...')

    # convert time string to datetime
    df[Columns.TIME] = df[Columns.TIME].apply(pd.to_datetime)

    # add features
    for unit in ['year', 'month', 'hour']:
        df[unit + Columns._TS_ADD] = df[Columns.TIME].apply(lambda t: getattr(t, unit))
    df[Columns.DAY] = df[Columns.TIME].apply(lambda t: t.weekday())
    df[Columns.POST_LEN] = df[Columns.TEXT].apply(lambda s: len(s.split()))
    df[Columns.DISTINCT] = df[Columns.TEXT].apply(lambda s: len(set(s.split())))
    df[Columns.AVG_WRD] = df[Columns.TEXT].apply(lambda s: np.mean([len(w) for w in s.split()]))
    df[Columns.MAX_WRD] = df[Columns.TEXT].apply(lambda s: np.max([len(w) for w in s.split()]))
    df[Columns.MIN_WRD] = df[Columns.TEXT].apply(lambda s: np.min([len(w) for w in s.split()]))
    df[Columns.NUMBERS] = df[Columns.TEXT].apply(lambda s: len(re.findall(r'\d+', s)))
    df[Columns.EX_MARKS] = df[Columns.TEXT].apply(lambda s: max([0] + [len(x) for x in re.findall(r'!+', s)]))

    # sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiments = pd.DataFrame(df[Columns.TEXT].apply(sia.polarity_scores).tolist())
    df = pd.concat([df, sentiments], axis=1)

    # split to train / test
    X_train, X_test, y_train, y_test = my_train_test_split(df, LABEL, 0.15)

    # get best k words for classification
    best_k_words = get_best_k(X_train[Columns.TEXT], y_train, k=20)

    ftr_words = set(best_k_words)

    # get most common words
    most_common_words = get_most_common_k(X_train[Columns.TEXT], k=20)

    ftr_words.update(most_common_words)

    # get ngrams
    ng = 2
    ngrams_voc = k_most_common_ngrams(X_train[Columns.TEXT], ng=ng, k=20)
    ftr_words.update(ngrams_voc)
    ch_ng = 3
    ch_ngrams_voc = k_most_common_char_ngrams(X_train[Columns.TEXT], ng=ch_ng, k=20)
    ftr_words.update(ch_ngrams_voc)

    custom_words = []
    ftr_words.update(custom_words)

    ftr_words = list(ftr_words)

    # vectorize ftr_words
    vectorizer = TfidfVectorizer(vocabulary=list(ftr_words),
                                 use_idf=True,
                                 ngram_range=(1, max(ng, ch_ng)))
    tfidf = vectorizer.fit_transform(X_train[Columns.TEXT])
    X_train = pd.concat([X_train, pd.DataFrame(tfidf.toarray(), columns=ftr_words)], axis=1)
    tfidf = vectorizer.transform(X_test[Columns.TEXT])
    X_test = pd.concat([X_test, pd.DataFrame(tfidf.toarray(), columns=ftr_words)], axis=1)

    print('Classifying...')
    features = [Columns.YEAR, Columns.MONTH, Columns.DAY, Columns.HOUR, Columns.POST_LEN, Columns.DISTINCT,
                Columns.AVG_WRD, Columns.MAX_WRD, Columns.MIN_WRD, Columns.NUMBERS, Columns.EX_MARKS]
    # scale features (DTs don't care, but other classifiers do)
    scaler = MinMaxScaler()
    X_train[features] = scaler.fit_transform(X_train[features])
    X_test[features] = scaler.transform(X_test[features])

    features += ftr_words + sentiments.columns.tolist()

    run_classifiers(X_train[features], y_train, X_test[features], y_test)

    ######################
    # analysis
    plt.figure()
    sns.catplot(y=LABEL, x=Columns.YEAR, data=df, kind='bar')
    plt.show()
    sns.catplot(y=LABEL, x=Columns.MONTH, data=df, kind='bar')
    sns.catplot(y=LABEL, x=Columns.DAY, data=df, kind='bar')
    plt.show()
    plt.figure()
    sns.scatterplot(y=df[LABEL], x=df[Columns.POST_LEN])
    plt.show()
    plt.figure()
    sns.scatterplot(y=df[LABEL], x=df[Columns.DISTINCT])
    plt.show()
