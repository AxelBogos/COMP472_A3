import pandas as pd
import numpy as np
import re
from NaiveBayesClassifier import *
def main():
    # Load data
    train = pd.read_csv('data/covid_training.tsv', sep='\t')
    test = pd.read_csv('data/covid_test_public.tsv', sep='\t',names=train.columns) #load it with the same column names

    # Drop useless columns & rename label
    train = train.drop(['q2_label', 'q3_label', 'q4_label', 'q5_label', 'q6_label', 'q7_label'], axis=1)
    test = test.drop(['q2_label', 'q3_label', 'q4_label', 'q5_label', 'q6_label', 'q7_label'], axis=1)
    # train.rename(columns={'q1_label': 'label'})
    # test.rename(columns={'q1_label': 'label'})

    # Pre-process the data. Lower-case, strip https links, tokenize.
    train['tokenized'] = train['text'].apply(lambda x: (preprocess(x)))

    # Get corpus (all the words in vocabulary)
    corpus = get_corpus(train['tokenized'])
    X_train = train['tokenized']
    y_train = train['q1_label']
  
    # Compute the Term Frequency Matrix use for training
    freqs_dict={}
    freqs_dictionnary(freqs_dict,X_train,y_train,corpus)

    #Initialize classifier and fit
    nbc = NaiveBayesClassifier(freqs_dict,smooth_factor)
    nbc.fit(X_train,y_train)

def compute_term_freq_matrix(X,corpus):
    '''
    Returns m x n term frequency matrix, where m is the vocabulary size and n is the number of instances in X.
    Element (i,j) is the frequency of word i in tweet j
    '''
    tfm = pd.DataFrame(index=corpus)
    for x in X:
        tfm = pd.concat([tfm, compute_tfm_column(x,corpus)], axis=1)
    return tfm

def compute_tfm_column(tweet, corpus, smooth_factor=0.01):
    '''
    Returns m x 1 column vector of the frequency of the ith word of the corpus in the tweet
    '''
    freqs=[(tweet.count(word)+smooth_factor)/(len(tweet)+(smooth_factor*len(corpus))) for word in corpus]
    return pd.DataFrame(freqs, index=corpus)

def freqs_dictionnary(result,X_train,y_train,corpus):
    for y, x in zip(y_train, X_train):
        for word in x:
            if word in corpus:
                # define the key, which is the word and label tuple
                pair = (word,y)

                # if the key exists in the dictionary, increment the count
                if pair in result:
                    result[pair] += 1

                # else, if the key is new, add it to the dictionary and set the count to 1
                else:
                    result[pair] = 1
    return result




def preprocess(text):
    '''
    Returns a tokenized version of the lower-cased text with https links stripped.
    Possible TODO: remove emojis as well from tweets
    '''
    text = text.lower()
    text= re.sub(r'http\S+', '', text) #remove https links strings
    return text.split()

def get_corpus(tokenized,filtered=False):
    '''
    Returns the corpus of all tweets. If filtered is true, only returns words that appear at least twice
    '''
    if filtered:
        return tokenized.explode().unique()[tokenized.explode().value_counts() > 1]
    return tokenized.explode().unique()

if __name__ == "__main__":
    main()