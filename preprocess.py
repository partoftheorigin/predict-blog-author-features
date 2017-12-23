import os
import re
import pandas as pd
import requests, zipfile, io
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

def download_dataset(url):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(os.getcwd() + '/Dataset')

# Filters the text inside the post tag in xml files and extracts the labels out of xml file name for Short text based classification.
# Processes the post text, removes stopwords, returns a pandas DataFrame containing label, text, gender, age, zodiac.
def process_data_short_text(folder_path):
    df = pd.DataFrame(columns=['label', 'text', 'gender', 'age', 'zodiac'])     # Empty DataFrame
    for filename in os.listdir(folder_path):

        # List of labels processed from file name.
        labels = filename.rstrip('.xml').lstrip('0123456789.').lower().split('.')

        # Beautiful soup to parse the xml files
        blog = BeautifulSoup(open(folder_path + '/' + filename), "lxml")

        for post in blog.find_all('post'):                  # Finds <post> tags inside xml
            post_text = post.get_text()                     # Fetches text inside <post> tags
            post_text = re.sub('[^A-Za-z]+', ' ', post_text).strip().lower().split()

            stop_words = set(stopwords.words('english'))    # Stopwords to remove from post text.
            post_text = [word for word in post_text if word not in stop_words]
            post_text = ' '.join(post_text)                 # Converts the list back to string.

            if len(post_text) > 0:
                df = df.append({'label': labels[2], 'text': post_text, 'gender': labels[0], 'age': labels[1], 'zodiac': labels[3]}, ignore_index=True)

    df.to_csv(os.getcwd() + 'preprocessed_data/blogdata_short_text.csv')                    # Write DataFrame to csv
    return df

# Filters the text inside the post tag in xml files and extracts the labels out of xml file name for long text based classification.
# Processes the post text, removes stopwords, returns a pandas DataFrame containing label, text, gender, age, zodiac.
def process_data_long_text(folder_path):
    df = pd.DataFrame(columns=['label', 'text', 'gender', 'age', 'zodiac'])  # Empty DataFrame

    # Unique File Names
    for f in os.listdir(folder_path):

        ds_gender = f.split('.')[1].lower()
        ds_age = f.split('.')[2]
        ds_label = f.split('.')[3].lower()
        ds_zodiac = f.split('.')[4].lower()

        blog_file = BeautifulSoup(open(folder_path + '/' + f), "lxml")

        pk = ''
        for post in blog_file.find_all('post'):
            pk = pk + post.text

        post_text = re.sub('[^A-Za-z]+', ' ', pk).strip().lower().split()
        stop_words = set(stopwords.words('english'))  # Stopwords to remove from post text.
        post_text = [word for word in post_text if word not in stop_words]
        post_text = ' '.join(post_text)  # Converts the list back to string.

        df = df.append({'label': ds_label, 'text': post_text, 'gender': ds_gender, 'age': ds_age, 'zodiac': ds_zodiac},ignore_index=True)

    # Save DataFrame
    df.to_csv(os.getcwd() + 'preprocessed_data/blogdata_long_text.csv')
    return df


if __name__ == "__main__":
    url = 'http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip'         # The Blog Authorship Corpus data set url
    download_dataset(url)                                           # Downloads the data set and saves it locally
    # Folder containing The Blog Authorship Corpus
    folder_path = os.getcwd() + 'Dataset/blogs'
    process_data_long_text(folder_path)
    process_data_short_text(folder_path)
