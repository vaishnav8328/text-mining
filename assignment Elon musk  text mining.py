# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 15:56:07 2022

@author: vaishnav
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import time
import string
import warnings
import spacy
from tqdm.notebook import tqdm_notebook

# for all NLP related operations on text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.classify import NaiveBayesClassifier
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud

# To identify the sentiment of text
from textblob import TextBlob

# ignoring all the warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


tweets = pd.read_csv(r"C:\anaconda\New folder (2)\Elon_musk.csv",encoding=('latin-1'))
tweets.drop(['Unnamed: 0'],inplace=True,axis=1)
tweets.rename({'Text':'Tweets'},axis=1,inplace=True)
tweets

##==============================================================================================================================================================================================================================================
#EDA
#==============================================================================================================================================================================================================================================

#no.of words
tweets['word_count'] = tweets['Tweets'].apply(lambda x: len(str(x).split(" ")))
tweets[['Tweets','word_count']].head()

#number of characters
tweets['char_count'] = tweets['Tweets'].str.len() ## this also includes spaces
tweets[['Tweets','char_count']].head()




#average word length
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

tweets['avg_word'] = tweets['Tweets'].apply(lambda x: avg_word(x))
tweets[['Tweets','avg_word']].head()


#Number of stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')

tweets['stopwords'] = tweets['Tweets'].apply(lambda x: len([x for x in x.split() if x in stop]))
tweets[['Tweets','stopwords']].head()

#Number of special characters
tweets['hashtags'] = tweets['Tweets'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
tweets[['Tweets','hashtags']].head()

#Number of numerics
tweets['numerics'] = tweets['Tweets'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
tweets[['Tweets','numerics']].head()

#Number of Uppercase words
tweets['upper'] = tweets['Tweets'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
tweets[['Tweets','upper']].head()


tweets.drop(['numerics','hashtags','stopwords','avg_word','char_count','word_count','upper'],axis=1,inplace=True)

#Number of Rare Words
freq = pd.Series(' '.join(tweets['Tweets']).split()).value_counts()[-10:]
freq


tweets.Tweets.str.contains('https://').value_counts() / len(tweets)




#Percentage of User Tags in the tweets
tweets.Tweets.str.contains('@').value_counts() / len(tweets)


# Total tweets
print('Total tweets this period:', len(tweets.index), '\n')
##==============================================================================================================================================================================================================================================
#for spelling corrections

from textblob import TextBlob
tweets['Tweets'][:5].apply(lambda x: str(TextBlob(x).correct()))



#N-grams
#N-grams are the combination of multiple words used together. Ngrams with N=1 are called unigrams. Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used.
#Unigrams do not usually contain as much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the language structure

TextBlob(tweets['Tweets'][3]).ngrams(1)
TextBlob(tweets['Tweets'][2]).ngrams(2)
TextBlob(tweets['Tweets'][5]).ngrams(3)

#==============================================================================================================================================================================================================================================

# Data Visualization
import itertools

c = list(
itertools.chain(
    *tweets.Tweets.map(lambda t: [handle.replace(":", "")[1:] for handle in t.split(" ") 
                            if '@' in handle.replace(":", "")]).tolist())
)

pd.Series(c).value_counts().head(20).plot.bar(
    figsize=(14, 7), fontsize=16, color='lightcoral'
)
plt.gca().set_title('@elonmusk top user tags', fontsize=20)
plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=45, ha='right', fontsize=16)
pass
#==============================================================================================================================================================================================================================================
#Test preprocessing

#==============================================================================================================================================================================================================================================

tweets=[Text.strip() for Text in tweets.Tweets] # remove both the leading and the trailing characters
tweets=[Text for Text in tweets if Text] # removes empty strings, because they are considered in Python as False
tweets[0:10]

# Joining the list into one string/text
tweets_text=' '.join(tweets)
tweets_text[:1000]


# remove Twitter username handles from a given twitter text. (Removes @usernames)
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
tweets_tokens=tknzr.tokenize(tweets_text)
print(tweets_tokens[:100])


# Again Joining the list into one string/text
tweets_tokens_text=' '.join(tweets_tokens)
tweets_tokens_text[:1000]


# Remove Punctuations 
no_punc_text=tweets_tokens_text.translate(str.maketrans('','',string.punctuation))
no_punc_text[:1000]


# remove https or url within text
import re
no_url_text=re.sub(r'http\S+', '', no_punc_text)
no_url_text[:1000]


import nltk.data
from nltk.tokenize import word_tokenize
text_tokens=word_tokenize(no_url_text)
print(text_tokens[:100])

# Tokens count
len(text_tokens)

#==============================================================================================================================================================================================================================================

# Remove Stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')

sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I','U0001F3B6','U0001F5A4']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens[:100])


# Making the Text in Lowercase
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[:100])


# Stemming 
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[:100])


# Lemmatization
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc[:100])

lemmas=[token.lemma_ for token in doc]
print(lemmas[:100])

clean_tweets=' '.join(lemmas)
clean_tweets[:1000]


#==============================================================================================================================================================================================================================================

#Generate Word Cloud
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=my_stop_words).generate(clean_tweets)
plot_cloud(wordcloud)

#==============================================================================================================================================================================================================================================
#For sentimental analysis


tweets1 = pd.read_csv(r"C:\anaconda\New folder (2)\Elon_musk.csv",encoding=('latin-1'))
tweets1.drop(['Unnamed: 0'],inplace=True,axis=1)
tweets1.rename({'Text':'Tweets'},axis=1,inplace=True)
tweets1

tweets1 = [Tweets.strip() for Tweets in tweets1.Tweets] # remove both the leading and the trailing characters
tweets1 = [Tweets for Tweets in tweets1 if Tweets] # removes empty strings, because they are considered in Python as False 




from nltk import tokenize
sentences = tokenize.sent_tokenize(" ".join(tweets1))
sentences[5:50]


text_df = pd.DataFrame(sentences, columns=['text'])
text_df


affinity_scores = text_df.set_index('text').to_dict() 
affinity_scores


#Custom function :score each word in a sentence in lemmatised form, 
#but calculate the score for the whole original sentence.
nlp = spacy.load('en_core_web_sm')
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score 

# test that it works
calculate_sentiment(text = 'tweets')

text_df['text'][:5].apply(lambda x: TextBlob(x).sentiment)


text_df['sentiment'] = text_df['text'].apply(lambda x: TextBlob(x).sentiment[0] )
text_df[['text','sentiment']]



import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(text_df['sentiment']) 


plt.figure(figsize=(15,10))
plt.xlabel('index')
plt.ylabel('sentiment')
sns.lineplot(data=text_df) 










