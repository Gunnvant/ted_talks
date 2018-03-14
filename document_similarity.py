# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:00:09 2018

@author: Gunnvant
"""
import os
import pandas as pd
import re
base_dir="E:\\Kaggle\\ted-talks"
os.chdir(base_dir)
transcripts=pd.read_csv("transcripts.csv")
transcripts['title']=transcripts['url'].map(lambda x:x.split("/")[-1])

## Extract key words using tfidf
pattern=re.compile(r'\d+')
transcripts['transcript'] = transcripts['transcript'].map(lambda x:re.sub(pattern,"",x))
from sklearn.feature_extraction import text
Text=transcripts['transcript'].tolist()
tfidf=text.TfidfVectorizer(input=Text,stop_words="english")
matrix=tfidf.fit_transform(Text)
matrix.shape

def get_imp_terms(x):
    x=x.todense()
    x=x.tolist()[0]
    x=pd.Series(x,index=tfidf.get_feature_names())
    x=x.sort_values(ascending=False)
    return x.head(4).index.tolist()
transcripts['imp_terms']=[get_imp_terms(x) for x in matrix]
transcripts['imp_terms_tfidf']=transcripts['imp_terms'].map(lambda x:",".join(x))
transcripts.drop("imp_terms",axis=1,inplace=True)

tfidf_bigrams=text.TfidfVectorizer(input=text,stop_words="english",ngram_range=(2,2),max_features=20000)
matrix_bigrams=tfidf_bigrams.fit_transform(Text)
matrix_bigrams.shape

def get_imp_terms_bigram(x):
    x=x.todense()
    x=x.tolist()[0]
    x=pd.Series(x,index=tfidf_bigrams.get_feature_names())
    x=x.sort_values(ascending=False)
    return x.head(4).index.tolist()

transcripts['imp_terms_bigrams']=[get_imp_terms_bigram(x) for x in matrix_bigrams]

transcripts['imp_terms_bigrams']=transcripts['imp_terms_bigrams'].map(lambda x:",".join(x))

## Keywords using RAKE, didn't work
import rake_nltk

r=rake_nltk.Rake()

def get_imp_terms_rake(x):
   r=rake_nltk.Rake()
   r.extract_keywords_from_text(x)
   r.get_ranked_phrases()[0:4]

transcripts['imp_terms_rake']=[get_imp_terms_rake(x) for x in Text]
transcripts['imp_terms_rake']=transcripts['imp_terms_rake'].map(lambda x:",".join(x))

transcripts.drop("imp_terms_rake",axis=1,inplace=True)

transcripts.to_csv("transcripts_key_words.csv",index=False,encoding="utf-8")

### Get Similarity Scores using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_unigram=cosine_similarity(matrix)
sim_bigram=cosine_similarity(matrix_bigrams)

def get_similar_articles(x):
    return ",".join(transcripts['title'].loc[x.argsort()[-5:-1]])

transcripts['similar_articles_unigram']=[get_similar_articles(x) for x in sim_unigram]    
transcripts['similar_articles_bigram']=[get_similar_articles(x) for x in sim_bigram]
