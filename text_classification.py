import re
import pandas as pd
import pickle, joblib
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def create_vector(df, name='cv'):
    # tf = Convert a collection of raw documents to a matrix of TF-IDF features.
    # cv = Convert a collection of text documents to a matrix of token counts

    if name == 'cv': vt = CountVectorizer(max_features=5000, stop_words='english')
    elif name == 'tf': vt = TfidfVectorizer(max_features=5000, stop_words='english')

    # cv.fit_transform() = Learn the vocabulary dictionary and return term-document matrix.
    # tf.fit_transform() = Learn vocabulary and idf, return term-document matrix.
    vector = vt.fit_transform(df.iloc[0:, 2].values).toarray()

    # pickle.dump(vt, open("Data/{}_vt.pkl".format(name), 'wb'))
    # joblib.dump(vector, open("Data/{}_vector.pkl".format(name), 'wb'))

    print('Vector returned.')
    return vt, vector


def classify(vector, df, name='mnb', col_num=1):
    if name == 'mnb':
        clf = MultinomialNB()
    elif name == 'bnb':
        clf = BernoulliNB()
    elif name == 'svc':
        clf = SVC()
    elif name == 'nvc':
        clf = NuSVC()
    elif name == 'lvc':
        clf = LinearSVC()
    elif name == 'lgr':
        clf = LogisticRegression()
    elif name == 'sgd':
        clf = SGDClassifier()
    elif name == 'rnf':
        clf = RandomForestClassifier(n_estimators=100)

    classifier = clf.fit(vector, df.iloc[0:, col_num].values)     # Fit classifier according to vector and input array.

    # pickle.dump(classifier, open("Data/{}_classifier_{}.pkl".format(name, col_num), 'wb'))

    return classifier


def accuracy(prediction, col_num=1):
    l = list()
    for i in range(0, len(prediction)):
        if test.iloc[i, col_num] == prediction[i]: l.append(1)
        else: l.append(0)
    acc_precentage = ((l.count(1)) / len(prediction)) * 100
    return acc_precentage


if __name__ == "__main__":
    # raw_text = input('Enter or paste text to get predictions: ')
    # clean_text = words = re.sub('[^A-Za-z]+', ' ', raw_text).strip().lower().split()

    for d in ['short', 'long']:
        df = pd.read_csv('/blogdata_{}_text.csv'.format(d))
        # df = pd.read_csv('/preprocessed_data/blogdata_short_text.csv')

        train, test = train_test_split(df, test_size=0.15)      # Split data into training and testing
        train = train.sample(frac=0.9, replace=True)            # Shuffle train data

        vt, vector = create_vector(train, name='cv')            # Convert into vectors to make computer understand

        # vt = pickle.load(open('Data/vttf.pkl', 'rb'))         # load vt from storage if already pickled
        # vector = joblib.load(open('Data/vectortf.pkl', 'rb')) # load vector from storage if already pickled

        # test_vector = vt.transform(clean_text)
        test_vector = vt.transform(test.iloc[0:, 2].values)   # Vector to test the model

        for col_name, i, algo in zip(['Label', 'Gender', 'Age', 'Zodiac'], [1, 3, 4, 5], ['mnb', 'mnb', 'mnb', 'mnb']):
            # Column Numbers: 1= Label, 3=Gender, 4=Age, 5=Zodiac

            model = classify(vector, train, name=algo, col_num=i)
            prediction = model.predict(test_vector)
            accuracy_precentage = accuracy(prediction, col_num=i)

            print('-----Author Info Prediction - {}: {}, Accuracy: {}, Algorithm: {}'.format(col_name, \
                    max(set(prediction), key=prediction.tolist().count), accuracy_precentage, algo))

    print('Program executed!')
