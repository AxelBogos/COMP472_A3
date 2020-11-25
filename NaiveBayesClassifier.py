import pandas as pd
import numpy as np
class NaivesBayesClassifier:
    def __init__(self, filtered):
        self.filtered = filtered
        pass

    def fit(self,X_train,y_train):
        self.corpus = get_corpus(X_train,self.filtered)
        self.tfm = compute_term_freq_matrix(X_train,self.corpus)
        pass

    def predict(self,X_test):
        print('to implement')
        pass
    
    def likelihood(self):
        print('to implement')
        pass

def compute_term_freq_matrix(X,corpus):
    '''
    Returns m x n term frequency matrix, where m is the number of instances in X and n is the vocabulary size.
    Element (i,j) is the frequency of word j in tweet i
    '''
    tfm = pd.DataFrame(columns=corpus)
    for x in X:
        result=pd.DataFrame(compute_tfm_column(x,corpus),columns=corpus)
        tfm = pd.concat([tfm, result], axis=0, ignore_index=True)
    return tfm

def compute_tfm_column(tweet, corpus, smooth_factor=0.01):
    '''
    Returns m x 1 column vector of the frequency of the ith word of the corpus in the tweet
    '''
    freqs=[(tweet.count(word)+smooth_factor)/(len(tweet)+(smooth_factor*len(corpus))) for word in corpus]
    return np.array(freqs).reshape((1,-1))

def get_corpus(train,filtered=False):
    '''
    Returns the corpus of all tweets. If filtered is true, only returns words that appear at least twice
    '''
    if filtered:
        return train.explode().unique()[train.explode().value_counts() > 1]
    return train.explode().unique()
