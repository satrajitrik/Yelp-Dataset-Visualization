from textblob import TextBlob
import json
import matplotlib.pyplot as plt
# matplotlib inline
from  matplotlib import style
import pandas as pd
from itertools import *
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
import textblob as tb
import time



t0=time.time()
categoreis=['Restaurants',
 'Food',
 'Nightlife',
 'Sandwiches',
 'Bars',
 'Mexican',
 'Fast Food',
 'Breakfast & Brunch',
 'Pizza',
 'Burgers',
 'Italian',
 'Coffee & Tea',
 'Salad']
for item in categoreis:
    filter_label=item.lower()
    final = [re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in item.lower().split("\n")]
    NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


    yelp = pd.ExcelFile('../data/final_review1.xlsx')
    df_all_review=yelp.parse('9.6K')
    #print(df_review.info())

    # filter review based on category
    filter_list=filter_label.split('&')

    df_review=pd.DataFrame()
    for i in range (len (filter_list)):
         temp=df_all_review[df_all_review['text'].str.contains(filter_list[i])]
         df_review=df_review.append(temp)


    review_list=df_review['text']
    #convert to lower
    review_list = [NO_SPACE.sub("", line.lower()) for line in review_list]
    review_list = [WITH_SPACE.sub(" ", line) for line in review_list]
    df_review['text']=review_list



    # remove special characters, numbers, punctuations
    # df_review['text'] =str(df_review['text'] )
    df_review['text'] = df_review['text'].str.replace("[^a-zA-Z#]", " ")


    #remove words less than 3 letters
    df_review['text'] = df_review['text'].apply(lambda x: ' '.join([word for word in x.split() if len(word)>3]))
    # print(df_review['text'].head())

    #remove stemming
    # from nltk.stem.porter import *
    # stemmer = PorterStemmer()
    # df_review['text'] =df_review['text'].apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
    #


    #extract only the sentences with the filter label from review text
    # df_review['text']=df_review['text'].apply(lambda text: [sent for sent in sent_tokenize(text)
    #                            if any([(w2.lower() in w.lower()) for w in wordpunct_tokenize(sent)
    #                                    for w2 in filter_label])
    #                                 ])
    # print(df_review['text'].head())


    #AFINN is list of English words a positive or negative integer assigned based on sentiment
    # To compute average sentiment score for each review
    # from afinn import Afinn
    # afinn = Afinn()
    # stop_words = set(stopwords.words('english'))
    # stop_words.update(['mr','.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation
    # unigrams = []
    # afinn_value = []
    # stars = []
    # for ind,review in islice(df_review.iterrows(),250000):
    #     df_review['text']= df_review['text'].fillna("")
    #     unigrams = ( [i.lower() for i in wordpunct_tokenize(df_review['text']) if i.lower() not in stop_words])
    #     afinn_value.append(np.mean(list(map(lambda x: afinn.score(str(x.encode('utf-8'))), unigrams))))
    #     stars.append(review['stars'])
    #
    # df_review['sentiment']=afinn_value
    #
    # print(df_review.head(5))


    #sentiment using textblob
    #textblob returns polarity, and subjectivity.
    #Polarity is float which lies in the range of [-1,1] where 1 is positive s and -1 is negative.
    #  Subjectivity refer to personal opinion, emotion etc
    # Subjectivity is also a float which lies in the range of [0,1]

    def sentiment_calculation(review_text):
        try:
            return TextBlob(review_text).sentiment
        except:
            return None


    df_review['total_sentiment'] = df_review['text'].apply(sentiment_calculation)
    df_review.sort_values('total_sentiment',inplace=True, ascending=False)


    # get positive and negative word corpus
    with open("../data/positive-words.txt", encoding="ISO-8859-1") as f:
     positive_words = f.read().split()[213:]

    with open("../data/negative-words.txt", encoding="ISO-8859-1") as f:
        negative_words = f.read().split()[213:]


    #get 3 words after and before category label and its sentiments
    # print(df_review.head())
    for index, row in df_review.iterrows():
         category=item.lower()
         words = re.findall(r'\w+', str(row['text']))
         matching = [s for s in words if category in s]

         filter_label=re.sub('\s+', '_', filter_label)
         df_review.at[index,filter_label+'_sentiscore']=0
         if  category in matching:
              for wordno, word in enumerate(words):
                   if(word== category):
                        indices = wordno
                        left_words = words[indices - 3:indices]
                        right_words = words[indices + 1:indices + 4]
                        positive_score = sum([t in positive_words for t in left_words])
                        positive_score  =positive_score+ sum([t in positive_words for t in right_words])
                        negative_score = sum([t in negative_words for t in left_words])
                        negative_score = negative_score+ sum([t in negative_words for t in right_words])
              net_score = positive_score - negative_score
              df_review.at[index,filter_label+'_sentiscore'] = net_score


    filter_label=re.sub('\s+', '_', filter_label)
    # df_review.sort_values('sentiscore',inplace=True, ascending=False)
    # df_review.to_csv('before_sum', sep='\t')
    # print(df_review.head())
    df_review[filter_label+'_total_score']=df_review[filter_label+'_sentiscore'].groupby(df_review['business_id']).transform('sum')
# df_review['score']=df_review.groupby(['business_id'])['sentiscore'].sum()
# df_review.sort_values('score',inplace=True, ascending=False)
    df_review.to_csv(item+'after_sum', sep='\t')
# print(df_review.groupby(['business_id']).head(100))
t1=time.time()
print(t1-t0)






