import os
import re
import codecs
import pandas as pd
import pickle, joblib
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# Filters the text inside the post tag in xml files and extracts the labels out of xml file name for Short text based classification.
# Processes the post text, removes stopwords, returns a pandas DataFrame containing label, text, gender, age, zodiac.
def process_data_short_text(folder_path):
    data = list()
    for filename in os.listdir(folder_path):

        # List of labels processed from file name.
        labels = filename.rstrip('.xml').lstrip('0123456789.').lower().split('.')

        # Beautiful soup to parse the xml files
        with codecs.open(folder_path + '/{}'.format(filename),
                         encoding='utf-8', errors='ignore') as fp: soup = BeautifulSoup(fp, "lxml")

        for post in soup.find_all('post'):                  # Finds <post> tags inside xml
            post_text = post.get_text()                     # Fetches text inside <post> tags
            post_text = re.sub('[^A-Za-z]+', ' ', post_text).strip().lower().split()

            stop_words = set(stopwords.words('english'))    # Stopwords to remove from post text.
            post_text = [word for word in post_text if word not in stop_words]
            post_text = ' '.join(post_text)                 # Converts the list back to string.

            if len(post_text) > 0:
                data.append([labels[2], post_text, labels[0], labels[1], labels[3]])

    df = pd.DataFrame(data, index=range(len(data)))         # Create a DataFrame from list of lists i.e. data.
    df.to_csv('Data/blogdata.csv')
    return df

# Filters the text inside the post tag in xml files and extracts the labels out of xml file name for long text based classification.
# Processes the post text, removes stopwords, returns a pandas DataFrame containing label, text, gender, age, zodiac.
def process_data_long_text(folder_path):
    clean_set = []
    
    # Dataset
    ds_pre = pd.DataFrame(columns=['label', 'text', 'gender', 'age', 'zodiac'])  # Empty DataFrame
    ls_train_files = os.listdir(folder_path) # List of Files in data folder
    num = 1
    
    # Unique File Names
    for f in ls_train_files:

        ds_gender = f.split('.')[1].lower()
        ds_age = f.split('.')[2]
        ds_label = f.split('.')[3].lower()
        ds_zodiac = f.split('.')[4].lower()

        if f.split('.')[3] not in data_labels:
            data_labels.add(f.split('.')[3].lower())

        print num, ' : ', f, ' : ', 19320-num
        num += 1
        blog_file = BeautifulSoup(open(loc + f), "lxml")

        pk = ''
        posts = blog_file.find_all('post')
        for post in posts:
            pk = pk + post.text

        letters = nltk.re.sub("[^a-zA-Z]", " ", pk)
        words = nltk.word_tokenize(letters)
        stop = set(stopwords.words("english"))
        rem_words = [w for w in words if not w in stop]

        ds_text = " ".join(rem_words)
        clean_set.append(ds_text)
        ds_pre = ds_pre.append({'label': ds_label, 'text': ds_text, 'gender': ds_gender, 'age': ds_age, 'zodiac': ds_zodiac},
                               ignore_index=True)

    # Save DataFrame
    ds_pre.to_csv('Data/blogdata_long.csv')
    return ds_pre
    
def create_vector(df, name='cv'):
    # tf = Convert a collection of raw documents to a matrix of TF-IDF features.
    # cv = Convert a collection of text documents to a matrix of token counts

    if name == 'cv': vt = CountVectorizer(max_features=5000)
    elif name == 'tf': vt = TfidfVectorizer(max_features=5000)

    # cv.fit_transform() = Learn the vocabulary dictionary and return term-document matrix.
    # tf.fit_transform() = Learn vocabulary and idf, return term-document matrix.
    vector = vt.fit_transform(df.iloc[0:, 2].values).toarray()

    pickle.dump(vt, open("Data/{}vt.pkl".format(name), 'wb'))
    joblib.dump(vector, open("Data/{}vector.pkl".format(name), 'wb'))

    print('Vector returned.')
    return vt, vector


def classify(vector, df, name='mnb', c=1):                  # c = Column Number
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

    classifier = clf.fit(vector, df.iloc[0:, c].values)     # Fit classifier according to vector and input array.
    print('Classifier trained.')

    pickle.dump(classifier, open("Data/{}classifier{}.pkl".format(name, c), 'wb'))

    return classifier


def accuracy(prediction, c=1):
    l = list()
    for i in range(0, len(prediction)):
        if test.iloc[i, c] == prediction[i]: l.append(1)
        else: l.append(0)
    acc_precentage = ((l.count(1)) / len(prediction)) * 100
    return acc_precentage

# Predict labels using raw text, features and model
def predict_category(text, features, model):
    clean_set = clean(text)

    features = vector.transform(clean_set)
    features = features.toarray()
    
    return model.predict(features)

# Clean text for model testting 
def clean(text):
    clean_set = []

    letters = nltk.re.sub("[^a-zA-Z]", " ", text)
    words = nltk.word_tokenize(letters)
    stop = set(stopwords.words("english"))
    rem_words = [w for w in words if not w in stop]
    retext = " ".join(rem_words)
    clean_set.append(retext)

    return retext

if __name__ == "__main__":
    folder_path = "/blogs"                                  # Path to dataset folder
    df = process_data_short_text(folder_path)               # Prepare blog data for short text classification
    #df_L = process_data_long_text(folder_path)                # Prepare blog data for long text classification
    train, test = train_test_split(df, test_size=0.15)      # Split data into training and testing
    train = train.sample(frac=0.9, replace=True)            # Shuffle train data

    vt, vector = create_vector(train, name='tf')            # Convert into vectors to make computer understand

    # vt = pickle.load(open('Data/vttf.pkl', 'rb'))
    # vector = joblib.load(open('Data/vectortf.pkl', 'rb'))

    test_vector = vt.transform(test.iloc[0:, 2].values)     # Vector to test the model

    acc = list()

    for i, algo in zip([1, 3, 4, 5], ['mnb', 'mnb', 'mnb', 'mnb']):
        # Column Numbers: 1= Label, 2=Text, 3=Gender, 4=Age, 5=Zodiac

        model = classify(vector, train, name=algo, c=i)
        prediction = model.predict(test_vector)
        ac = accuracy(prediction, c=i)

        print(ac, i, algo)

    print('Program executed!')