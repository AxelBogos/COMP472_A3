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
    train = train.rename(columns={'q1_label': 'label'})
    test = test.rename(columns={'q1_label': 'label'})

    # Pre-process the data. Lower-case, strip https links, tokenize.
    train['tokenized'] = train['text'].apply(lambda x: (preprocess(x)))


    # Initialize classifier and fit
    nbc = NaivesBayesClassifier(filtered=False)
    nbc.fit(train['tokenized'],train['label'])


def preprocess(text):
    '''
    Returns a tokenized version of the lower-cased text with https links stripped.
    Possible TODO: remove emojis as well from tweets
    '''
    text = text.lower()
    text= re.sub(r'http\S+', '', text) #remove https links strings
    return text.split()


if __name__ == "__main__":
    main()