import numpy as np
class NaiveBayesClassifier:
    def __init__(self, freqs,smooth_factor):
        self.freqs=freqs
        self.smooth_factor=smooth_factor
        self.log_prior=0
        self.log_likelihood={}

    def fit(self,X_train,y_train):
        '''
        Input:
            freqs: dictionary from (word, label) to how often the word appears
            train_x: a list of tweets
            train_y: a list of labels correponding to the tweets (No,Yes)
        '''
        # calculate V, the number of unique words in the vocabulary
        vocab = set([pair[0] for pair in self.freqs.keys()])
        V = len(vocab)

        # calculate N_pos and N_neg
        N_pos = N_neg = 0
        for pair in self.freqs.keys():
            # if the label is positive (greater than zero)
            if pair[1] > 0:

                # Increment the number of positive words by the count for this (word, label) pair
                N_pos +=self.freqs[pair] 

            # else, the label is negative
            else:

                # increment the number of negative words by the count for this (word,label) pair
                N_neg +=self.freqs[pair]

        # Calculate D, the number of documents
        D = len(y_train)

        # Calculate D_pos, the number of positive documents (*hint: use sum(<np_array>))
        D_pos = sum(y_train)

        # Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
        D_neg = D-D_pos

        # Calculate logprior
        self.log_prior = np.log(D_pos)-np.log(D_neg)

        # For each word in the vocabulary...
        for word in vocab:
            # get the positive and negative frequency of the word
            freq_pos = self.freqs.get((word,1),0)
            freq_neg = self.freqs.get((word,0),0)

            # calculate the probability that each word is positive, and negative
            p_w_pos = (freq_pos+self.smooth_factor)/(N_pos+V*self.smooth_factor)
            p_w_neg = (freq_neg+self.smooth_factor)/(N_neg+V*self.smooth_factor)

            # calculate the log likelihood of the word
            self.log_likelihood[word] = np.log(p_w_pos)-np.log(p_w_neg)

       
       

    def predict(self,X_test):
        pass
    
    def likelihood(self):
        pass