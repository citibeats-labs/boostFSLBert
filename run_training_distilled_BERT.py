from os import getcwd
from os.path import join

import pickle
import pandas as pd

PATH_REPO = getcwd()
PATH_DATA = join(PATH_REPO, 'data')
PATH_MODELS = join(PATH_REPO, 'models')

from utils.train import *

import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='train model')
	
	parser.add_argument('--PATH_DATA',   metavar='d', type=str, help='PATH to data folder'  , default=PATH_DATA)
	parser.add_argument('--PATH_MODELS', metavar='m', type=str, help='PATH to models folder', default=PATH_MODELS)
	parser.add_argument('--name', metavar='name', type=str, help='project name')
	parser.add_argument('--bert', metavar='b', type=str, help='type of model from HugginFace', default='distilbert-base-uncased')
	parser.add_argument('--max_length', metavar='l', type=int, help='max length for the tokenizer', default=64)
	parser.add_argument('--epochs', metavar='e', type=int, help='epochs', default=30)
	parser.add_argument('--bs', metavar='bs', type=int, help='batch_size', default=16)
	parser.add_argument('--lr', metavar='lr', type=float, help='learning rate for the classifier', default=2e-05)
	parser.add_argument('--do', metavar='do', type=float, help='dropout rate', default=0.5)
	parser.add_argument('--alpha', metavar='a', type=float, help='alpha for disitllation loss', default=0.5)
	parser.add_argument('--T', metavar='T', type=float, help='Temperature for disitllation loss', default=5)


	args = parser.parse_args()

	X, y = get_X_y(args.PATH_DATA, args.name, 'prob_hatred')

	tokenizer, transformer_model = load_transformer_models(args.bert)

	model = train_distilled_model(X, y, tokenizer, 
				       transformer_model, categories, args.do, args.lr, args.epochs,
				       args.bs, args.max_length, args.alpha, args.T)
	  
	model.save_weights(join(args.PATH_MODELS, f'{args.name}_distilled.h5'))