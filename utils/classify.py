import pickle

from os.path import join, isfile
import numpy as np
import tensorflow as tf
from transformers import pipeline
from train import train_projection, get_model, get_inputs
from pre_process import load_transformer_models


##
## CitiFit v0.1: scrapping data on twitter and training classifier
##

def classify_CitiFit_0_1(PATH_MODELS, name, X):

	clf = pickle.load(open(join(PATH_MODELS, 'model_{}.pkl'.format(name)), 'rb'))
	preds = clf.predict(X)
	probas = clf.predict_proba(X)

	return preds, probas

##
## FSL: BERT application
##


##
## ZSL: PET approach
##

def classify_PET(X, PATH_MODELS, bert, special_tokens, max_length, verbalizers, name_model, p=0.5):
	"""
	Objective: classify sentences with the PET classifier

	Inputs:
		- X, np.array: the sentences
		- PATH_MODELS, str: the path to the models folder
		- bert, str: the name of models look at https://huggingface.co/models for all models
		- special_tokens, list: list of str, where they are tokens to be considered as one token
		- max_length, int: the maximum length for the tokenizer
		- verbalizers, list: the class of the model
		- name_model, str: the model name in PATH_MODELS
		- p, float: the threshold to consider if the probability is 1 or 0
	Outputs:
		- predictions, np.array: the predictions of the model
		- similarities, np.array: the probabilities associated to each class
	"""
	tokenizer, transformer_model = load_transformer_models(bert, special_tokens)

	model = get_model(max_length, transformer_model, num_labels=len(verbalizers), 
					  name_model=join(PATH_MODELS, '{}.h5'.format(name_model)))

	inputs = tokenizer.batch_encode_plus(X, add_special_tokens=True, max_length=max_length, 
										 padding='max_length',  return_attention_mask=True,
										 return_token_type_ids=True, truncation=True)

	X = [np.asarray(inputs['input_ids'], dtype='int32'), np.asarray(inputs['attention_mask'], dtype='int32')]

	similarities = model.predict(X)

	predictions = get_predictions(similarities, p=p)

	return predictions, similarities



##
## ZSL: embeddings approach
##

def classify_ZSL(X, labels_encoded, PATH_MODELS, PATH_CONFIG, config_file, name, w2v_model, tokenizer, transformer_model, k=100):

	proj = get_projections(PATH_MODELS, PATH_CONFIG, config_file, name,
							 w2v_model, tokenizer, transformer_model, k=100)

	X = proj.predict(X)
	labels_encoded = {label: proj.predict(value.reshape(1,-1)) for label, value in labels_encoded.items()}
	
	similarities = get_similarities(X, labels_encoded)
	predictions = get_predictions(similarities, p=0.90)
	
	return predictions, similarities


##
## ZSL: embeddings approach - without projection
##

def classify_CitiFit_0(X, labels_encoded):
	
	similarities = get_similarities(X, labels_encoded)
	predictions = get_predictions(similarities, p=0.90)
	
	return predictions, similarities


def get_projections(PATH_MODELS, PATH_CONFIG, config_file, name, w2v_model, tokenizer, transformer_model, k=100):
	"""
	Objective: Get the projection from bert to w2v already learned or create it
	
	Inputs:
		- PATH_MODELS, str: the path to the models folder
		- PATH_CONFIG, str: the path to the config folder
		- config_file, dict: the configuration files
		- name, str: name to save the model
		- w2v_model, str: the name of the w2v model
		- tokenizer, transformers.tokenization_distilbert.DistilBertTokenizer: the tokenizer of the model
		- transformer_model, transformers.modeling_tf_distilbert.TFDistilBertModel: the transformer model that
		- k, int: the number of data to make the regression

	Outputs:
		- proj, sklearn.linear_model._ridge.Ridge: the projections from bert to w2v
	"""

	PATH = join(PATH_MODELS, 'projection_{}.pkl'.format(name))

	if isfile(PATH):
		proj = pickle.load(open(PATH, 'rb'))
	
	else:
		train_projection(PATH_MODELS, PATH_CONFIG, config_file, name, 
						 w2v_model, tokenizer, transformer_model, k=k) 
			
		proj = pickle.load(open(PATH, 'rb'))
	
	return proj


def get_similarities(sentences_encoded, labels_encoded):
	"""
	Objective: get the similarities of sentences with the labels depending on their encoding
	
	Inputs:
		- sentences_encoded, np.array: encoded sentences
		- labels_encoded, dict: the encoded labels dictionary
	Outputs:
		- similarities, np.array: the similarities between sentences encoded and labels_encoded
	"""
	sim = {}
	for lab, lab_encoded in labels_encoded.items():
		sim[lab] = np.array(-tf.keras.losses.cosine_similarity(sentences_encoded, lab_encoded)).reshape(-1, 1)
	
	labels = list(labels_encoded.keys())
	labels.sort()
	similarities = np.concatenate(([sim.get(label) for label in labels]), axis=1)

	return similarities

##
## NLI: loading a classifier NLI
##


def classify_NLI(X, candidate_labels):

	classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
	probas = nli_classification(X, candidate_labels, classifier)
	preds = get_predictions(probas, p=0.5)

	return preds, probas


def nli_classification(sentences, candidate_labels, classifier):
	"""
	Objective: get the probabilities of each sentence regarding each candidate_label
	
	Inputs:
		- sentences, np.array: sentences to classify
		- candidate_labels. list: the list of potential labels
	Outputs:
		- probabilities, np.array: shape(length(sentences), len(labels)) with probability to be associated to each
									label
	"""
	outputs = classifier(sentences, candidate_labels, 
						 hypothesis_template='This text is about {}.', multi_class=True)
	
	probabilities = np.array([order_probabilities(x) for x in outputs])

	return probabilities


def order_probabilities(result):
	"""
	Objective: order the probabilities according to the labels alphabetic order
	
	Inputs:
		- results, dict: the dictionary of the results for the sentence
	Outputs:
		- output, np.array: the array of probabilities ordered by label alphabetic order
	"""
	probs = dict(zip(result.get('labels'), result.get('scores')))
	labels = result.get('labels')
	labels.sort()
	output = np.array([probs.get(label) for label in labels])
	
	return output


def get_predictions(probabilities, p=0.5):
	"""
	Objective: from probabilities or similarities get the predictions depending on the threshold we put
	
	Inputs:
		- probabilities, np.array: probabilities
		- p, float: probability threshold
	Outputs:
		- preds np.array: same shape of proababilities but with predictions only
	"""
	
	preds = np.sign((probabilities > p) * probabilities)

	return preds