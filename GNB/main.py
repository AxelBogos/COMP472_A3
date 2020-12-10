import re
import pandas as pd
from GNB import *
from GNB.NaiveBayesClassifier import NaivesBayesClassifier


def main():
    # Load data
    train = pd.read_csv('data/covid_training.tsv', sep='\t')
    test = pd.read_csv('data/covid_test_public.tsv', sep='\t', names=train.columns) #load it with the same column names

    # Drop useless columns & rename label
    train = train.drop(['q2_label', 'q3_label', 'q4_label', 'q5_label', 'q6_label', 'q7_label'], axis=1)
    test = test.drop(['q2_label', 'q3_label', 'q4_label', 'q5_label', 'q6_label', 'q7_label'], axis=1)
    train = train.rename(columns={'q1_label': 'label'})
    test = test.rename(columns={'q1_label': 'label'})

    # Pre-process the data. Lower-case, strip https links, tokenize.
    train['tokenized'] = train['text'].apply(lambda x: (preprocess(x)))
    test ['tokenized'] = test ['text'].apply(lambda x: (preprocess(x)))
    # Convert string to bool: label->Ground Truth
    train['GT'] = train['label'].apply(lambda x:(to_bool(x)))
    test['GT'] = test['label'].apply(lambda x:(to_bool(x)))


    # Initialize classifier and fit
    nbc = NaivesBayesClassifier(filtered=False)
    nbc.fit(train['tokenized'],train['GT'])
    nbc.predict(test['tokenized'],test['GT'],test['tweet_id'], analyse=True, prior=False)
    nbc = NaivesBayesClassifier(filtered=True)
    nbc.fit(train['tokenized'], train['GT'])
    nbc.predict(test['tokenized'], test['GT'], test['tweet_id'], analyse=True, prior=False)




def to_bool(text):
    if text=='no':
        return False
    else:
        return True


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