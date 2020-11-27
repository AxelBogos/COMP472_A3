import pandas as pd
import numpy as np
import sklearn.metrics as skm
class NaivesBayesClassifier:
    def __init__(self, filtered,smoothing=0.01):
        self.filtered = filtered
        self.smoothing = smoothing
        self.log_prior = 0
        self.tfm = {}
        self.log_likelihood = {}
        self.corpus=None
      

    def fit(self,X_train,y_train):

        self.corpus = get_corpus(X_train,self.filtered)
        freqs_dictionnary(self.tfm,X_train,y_train,self.corpus)
        # Convert to numpy array
        X_train=np.asarray(X_train)
        y_train=np.asarray(y_train,dtype='bool')


        # calculate N_pos and N_neg
        N_pos = N_neg = 0
        for pair in self.tfm.keys():
            # if the label is positive
            if pair[1]:
                # Increment the number of positive words by the count for this (word, label) pair
                N_pos +=self.tfm[pair] 
            # else, the label is negative
            else:
                # increment the number of negative words by the count for this (word,label) pair
                N_neg +=self.tfm[pair]

        #nb of documents
        D = len(y_train)
        #nb of positive documents
        D_pos = np.sum(y_train)
        #nb of negative documents
        D_neg = D-D_pos

        # Calculate logprior
        self.log_prior = np.log10(D_pos)-np.log10(D_neg)

        # For each word in the corpus
        for word in self.corpus:
            # get the positive and negative frequency of the word
            freq_pos = self.tfm.get((word,1),0)
            freq_neg = self.tfm.get((word,0),0)

            # calculate the probability that each word is positive, and negative
            p_w_pos = (freq_pos+self.smoothing)/(N_pos+ len(self.corpus)*self.smoothing)
            p_w_neg = (freq_neg+self.smoothing)/(N_neg+ len(self.corpus)*self.smoothing)

            # calculate the log likelihood of the word
            self.log_likelihood[word] = np.log10(p_w_pos)-np.log10(p_w_neg)

    def predict(self,X_test,y_test,tweet_id,analyse=False,prior=True):
        if analyse:
            print('Truth Predicted Tweet')
        
        predictions=[]
        scores=[]
        for tweet, y in zip(X_test,y_test):
            # initialize probability to the log prior probability
            if prior:
                p = self.log_prior
            else:
                p=0
            for word in tweet:
                # check if the word exists in the loglikelihood dictionary
                if word in self.log_likelihood:
                    # add the log likelihood of that word to the probability
                    p += self.log_likelihood[word]
            scores.append(p)
            #if probability >0 the Tweet is classified as Verified
            if p>0:
                predictions.append(True)
            else:
                predictions.append(False)

            #Print the errors
            if analyse:
                if y != (p > 0):
                    print('{:d}\t{:.2f}\t{:s}'.format(y, p > 0,' '.join(tweet)))

        # Calculate Metrics
        accuracy=skm.accuracy_score(y_test,predictions)
        precisions=skm.precision_score(y_test,predictions),skm.precision_score(y_test,predictions,pos_label=0)
        recalls=skm.recall_score(y_test,predictions),skm.recall_score(y_test,predictions,pos_label=0)
        f1s=skm.f1_score(y_test,predictions) ,skm.f1_score(y_test,predictions,pos_label=0)

        #Write to files
        self.create_Trace_file(tweet_id,predictions,y_test,scores)
        self.create_Eval_file(accuracy,precisions,recalls,f1s)

    def create_Trace_file(self,ids,predictions,Truths,scores):
        if self.filtered:
            path='trace_NB-BOW-FV.txt'
        else:
            path='trace_NB-BOW-OV.txt'
        with open(path,'+w') as f:
            for i in range(len(ids)):
                pred='yes' if predictions[i] else 'no'
                truth='yes' if Truths[i] else 'no'
                result='correct' if pred==truth else 'wrong'
                f.write('{:d}  {:s}  {:.2f}  {:s}  {:s}\n'.format(ids[i],pred,scores[i],truth,result))

    def create_Eval_file(self,acc,precisions,recalls,f1s):
        if self.filtered:
            path='eval_NB-BOW-FV.txt'
        else:
            path='eval_NB-BOW-OV.txt'
        with open(path,'+w') as f:
            f.write('{:.4f}\n'.format(acc))
            f.write('{:.4f}  {:.4f}\n'.format(precisions[0],precisions[1]))
            f.write('{:.4f}  {:.4f}\n'.format(recalls[0],recalls[1]))
            f.write('{:.4f}  {:.4f}'.format(f1s[0],f1s[1]))

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
    

def get_corpus(train,filtered=False):
    '''
    Returns the corpus of all tweets. If filtered is true, only returns words that appear at least twice
    '''
    if filtered:
        return train.explode().unique()[train.explode().value_counts() > 1]
    return train.explode().unique()
