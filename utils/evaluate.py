import pandas as pd
from os.path import join
import ast
from collections import Counter

import logging

logging.basicConfig(level=logging.INFO)



def evaluate(y_true, y_preds):
	"""
	Objective: get the evaluations metrics from the predictions
	Inputs:
		- y_true, np.array: true intents
		- y_preds, np.array: predictions of the model
	Outputs:
		- metrics, dict: a dictionary with the precision, recall accuracy and f1
	"""
	metrics = {'tn': 0, 'fn': 0, 'tp': 0, 'fp': 0}
	
	metrics['accuracy'] = round((y_true == y_preds).mean() * 100, 2)
	
	for i, pred in enumerate(y_preds):
		true = y_true[i]
		if (true, pred) == (0, 0):
			metrics['tn'] += 1
		elif (true, pred) == (1, 1):
			metrics['tp'] += 1
		elif (true, pred) == (0, 1):
			metrics['fp'] += 1
		else:
			metrics['fn'] += 1
	
	num = metrics['tp']
	denum_r = metrics['tp'] + metrics['fn']
	denum_p = metrics['tp'] + metrics['fp']
			
	r = round(num / denum_r * 100, 2) if denum_r > 0 else 0
	p = round(num / denum_p * 100, 2) if denum_p > 0 else 0
	
	metrics['precision'] = p
	metrics['recall'] = r
	metrics['f1'] = 2 * r * p / (r + p) if (r + p) > 0 else 0
	
	return metrics


def seq_ngrams(xs, n, stop_words):
    """
    Objective: list all the n-grams of a sequence
    Inputs:
        - xs, list: list of the tokens
        - n, int: the level of grams we want
        - stop_words, list: the list of stop words we don't want to have alone in the tokens
    Outputs:
        - ngrams, list: list of n-grams 
    """

    ngrams = [' '.join(xs[i:i+n]).replace(' ##', '') for i in range(len(xs)-n+1) 
              if len(xs[i:i+n]) > 1 or xs[i:i+n][0] not in stop_words]

    return ngrams


def get_counts_ngrams(texts, tokenizer, n=6, stop_words=None):
	"""
	Objective: get the count of ngrams in the texts from any tokenizer (here HF's one)
	Inputs:
		- texts, np.array: the texts we want to extract ngrams from
		- tokenizer, transformers.tokenization_distilbert.DistilBertTokenizer: the tokenizer of the model
		- n, int: the maximum length for ngrams
		- stop_words, list: the stop words to avoid alone
	Outputs:
		- ngrams_counts, dict(Counter.object): the counts of ngrams for the texts depending the tokenizer
	"""
	tokens = [tokenizer.tokenize(x) for x in texts]

	ngrams = []

	for _tokens in tokens:
	    for i in range(1, n):
	        ngrams += seq_ngrams(_tokens, i, stop_words)

	ngrams_counts = Counter(ngrams)

	return dict(ngrams_counts)