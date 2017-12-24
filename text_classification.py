import re
import os
import pandas as pd
import pickle, joblib
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from preprocess import download_dataset, process_data_short_text, process_data_long_text


def create_vector(df, vectorizer):
    # tf = Convert a collection of raw documents to a matrix of TF-IDF features.
    # cv = Convert a collection of text documents to a matrix of token counts

    if vectorizer == 'cv': vt = CountVectorizer(max_features=50, stop_words='english')
    elif vectorizer == 'tf': vt = TfidfVectorizer(max_features=500, stop_words='english')

    # cv.fit_transform() = Learn the vocabulary dictionary and return term-document matrix.
    # tf.fit_transform() = Learn vocabulary and idf, return term-document matrix.
    vector = vt.fit_transform(df.iloc[0:, 1].values).toarray()

    pickle.dump(vt, open("{}_vt.pkl".format(vectorizer), 'wb'))
    joblib.dump(vector, open("{}_vector.pkl".format(vectorizer), 'wb'))

    print('Vector returned.')
    # return vt, vector


def classify(vector, df, algorithm, column):
    if algorithm == 'mnb':
        clf = MultinomialNB()
    elif algorithm == 'bnb':
        clf = BernoulliNB()
    elif algorithm == 'svc':
        clf = SVC()
    elif algorithm == 'nvc':
        clf = NuSVC()
    elif algorithm == 'lvc':
        clf = LinearSVC()
    elif algorithm == 'lgr':
        clf = LogisticRegression()
    elif algorithm == 'sgd':
        clf = SGDClassifier()
    elif algorithm == 'rnf':
        clf = RandomForestClassifier(n_estimators=100)

    # Fit classifier according to vector and input array.
    classifier = clf.fit(vector, df.iloc[0:, column].values)

    # pickle.dump(classifier, open("Data/{}_classifier_{}.pkl".format(algorithm, column), 'wb'))

    return classifier


def accuracy(prediction, column):
    l = list()
    for i in range(0, len(prediction)):
        if test.iloc[i, column] == prediction[i]: l.append(1)
        else: l.append(0)
    accuracy_percentage = ((l.count(1)) / len(prediction)) * 100
    return accuracy_percentage


if __name__ == "__main__":
    # The Blog Authorship Corpus data set url
    url = 'http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip'

    # Downloads the data set and saves it locally
    download_dataset(url)

    # Folder containing The Blog Authorship Corpus
    folder_path = os.getcwd() + '/Dataset/blogs'

    # process_data_short_text(folder_path)
    process_data_long_text(folder_path)

    # raw_text = input('Enter or paste text to get predictions: ')
    # clean_text = words = re.sub('[^A-Za-z]+', ' ', raw_text).strip().lower().split()

    # df = pd.read_csv('blogdata_long_text.csv')
    # df = pd.read_csv('/preprocessed_data/blogdata_short_text.csv')

    # Split data into training and testing
    train, test = train_test_split(pd.read_csv('blogdata_long_text.csv'), test_size=0.15)

    # Shuffle train data
    train = train.sample(frac=0.9, replace=True)

    # Convert into vectors to make computer understand
    create_vector(train, vectorizer='cv')

    # load vt from storage if already pickled
    vt = pickle.load(open('cv_vt.pkl', 'rb'))

    # load vector from storage if already pickled
    vector = joblib.load(open('cv_vector.pkl', 'rb'))

    # test_vector = vt.transform(clean_text)

    # Vector to test the model
    test_vector = vt.transform(test.iloc[0:, 1].values)

    for col_name, i, algo in zip(['Label', 'Gender', 'Age', 'Zodiac'], [0, 2, 3, 4], ['mnb', 'mnb', 'mnb', 'mnb']):
        # Column Numbers: 0= Label, 2=Gender, 3=Age, 4=Zodiac

        model = classify(vector, train, algorithm=algo, column=i)
        prediction = model.predict(test_vector)
        accuracy_percentage = accuracy(prediction, column=i)

        print('-----Author Info Prediction - {}: {}, Accuracy: {}, Algorithm: {}'.format(col_name, max(set(prediction), key=prediction.tolist().count), accuracy_percentage, algo))

    print('Program executed!')
