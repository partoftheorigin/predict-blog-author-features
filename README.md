# Contextual Classification of Long & Short Text

[![Build Status](https://travis-ci.org/partoftheorigin/Contextual_Classification_of_Long_and_Short_Text.svg?branch=master)](https://travis-ci.org/partoftheorigin/Contextual_Classification_of_Long_and_Short_Text) [![Coverage Status](https://coveralls.io/repos/github/partoftheorigin/Contextual_Classification_of_Long_and_Short_Text/badge.svg?branch=master)](https://coveralls.io/github/partoftheorigin/Contextual_Classification_of_Long_and_Short_Text?branch=master) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](README.md#pull-requests)

Predicts gender, age, label, and zodiac sign of the author from the given text.

## Pretrained Model for Long Text Classification
* Predicts text label for long paragraph of text. 
* Prepare features(vector) on this dataset from existing functions and use the model for label prediction, model is trained on MultinomialNB.
* Download Link: https://www.dropbox.com/s/109c4ccx9fjkebc/text_class_model?dl=0

## Algorithms
* [Multinomial Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes)
* [Logistic Regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
* [Bernoulli Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes)
* [Random Forests](http://scikit-learn.org/stable/modules/ensemble.html#forest)
* [Stochastic Gradient Descent](http://scikit-learn.org/stable/modules/sgd.html#sgd)

### Accuracy
Algorithm            | Label               | Gender              | Age                 | Zodiac
-------------------- | ------------------  | ------------------- | ------------------  | --------------------
Logistic Regression  | tf-0.367, cv-0.371* | tf-0.527, cv-0.571* | tf-0.145, cv-0.148  | tf-0.095, cv-0.101*
MultinomialNB        | tf-0.367, cv-0.354  | tf-0.568, cv-0.570  | tf-0.149*, cv-0.137 | tf-0.099, cv-0.101*
Random Forests       | tf-0.354, cv-0.326  | tf-0.520, cv-0.560  | tf-0.126, cv-0.141  | tf-0.094, cv-0.096
BernoulliNB          | tf-0.322, cv-0.322  | tf-0.556, cv-0.556  | tf-0.116, cv-0.116  | tf-0.099, cv-0.099
SGDClassifier        | tf-0.361, cv-0.334  | tf-0.504, cv-0.563  | tf-0.073, cv-0.072  | tf-0.076, cv-0.079

tf = TfidfVectorizer, cv=CountVectorizer, values represent accuracy score.

## Dataset
### The Blog Authorship Corpus
The Blog Authorship Corpus consists of the collected posts of 19,320 bloggers gathered from blogger.com in August 2004. The corpus incorporates a total of 681,288 posts and over 140 million words - or approximately 35 posts and 7250 words per person.  

Each blog is presented as a separate file, the name of which indicates a blogger id# and the blogger’s self-provided gender, age, industry and astrological sign. (All are labeled for gender and age but for many, industry and/or sign is marked as unknown.)

All bloggers included in the corpus fall into one of three age groups:
* 8240 "10s" blogs (ages 13-17)
* 8086 "20s" blogs(ages 23-27)
* 2994 "30s" blogs (ages 33-47)

For each age group there are an equal number of male and female bloggers.   

Each blog in the corpus includes at least 200 occurrences of common English words. All formatting has been stripped with two exceptions. Individual posts within a single blogger are separated by the date of the following post and links within a post are denoted by the label urllink.

Download Corpus: http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip


#### Dependencies
* [Python 3.6](https://www.python.org)
* [scikit-learn](http://scikit-learn.org)
* [pandas](https://pandas.pydata.org)
* [joblib](https://pypi.python.org/pypi/joblib)
* [nltk](https://pypi.python.org/pypi/nltk)
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

## Pull requests
We hope that other people can benefit from the project. We are thankful for any contributions from the community.
