# Contextual Classification of Short Text
Predicts gender, age, label, and zodiac sign of the author from the given text.

## Algorithms
* [Multinomial Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes)
* [Logistic Regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
* [Bernoulli Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes)
* [Random Forests](http://scikit-learn.org/stable/modules/ensemble.html#forest)
* [Stochastic Gradient Descent](http://scikit-learn.org/stable/modules/sgd.html#sgd)

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
