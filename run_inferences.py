from os import getcwd
from os.path import join

import pandas as pd

PATH_REPO = getcwd()
PATH_DATA = join(PATH_REPO, 'data')
PATH_MODELS = join(PATH_REPO, 'models')

from utils.train import get_model, load_transformer_models
from utils.pre_process import pre_process
from utils.inferences import get_inputs

import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='apply model inferences')
	

	parser.add_argument('--PATH_DATA',   metavar='d', type=str, help='PATH to data folder'  , default=PATH_DATA)
	parser.add_argument('--PATH_MODELS', metavar='m', type=str, help='PATH to models folder', default=PATH_MODELS)
	parser.add_argument('--name_model', metavar='nm', type=str, help='project name')
	parser.add_argument('--name_data', metavar='nd', type=str, help='dataset name')
	parser.add_argument('--bert', metavar='b', type=str, help='type of model from HugginFace', default='distilbert-base-uncased')
	parser.add_argument('--max_length', metavar='l', type=int, help='max length for the tokenizer', default=64)
	parser.add_argument('--lr', metavar='lr', type=float, help='learning rate for the classifier', default=2e-05)
	parser.add_argument('--do', metavar='do', type=float, help='dropout rate', default=0.5)
	parser.add_argument('--i', metavar='al', type=int, help='number of models', default=0)
	parser.add_argument('--english', metavar='e', type=bool, help='english or not', default=False)
	parser.add_argument('--text_var', metavar='t', type=str, help='variable name for text', default='text')

	args = parser.parse_args()

	df = pd.read_csv(join(args.PATH_DATA, f'{args.name_data}.csv'), engine='python', sep=';')
	df = pre_process(df, text_var=args.text_var, english=args.english)
	
	tokenizer, transformer_model = load_transformer_models(args.bert, [])
	
	model = get_model(args.max_length, transformer_model, num_labels=1, rate=args.do, 
										name_model=join(args.PATH_MODELS, f'{args.name_model}.h5'))
	
	X = df.loc[:, 'text_pp'].values
	
	inputs_X = get_inputs(tokenizer, X, args.max_length)
	
	y_probas = model.predict(inputs_X)
	
	probabilities = pd.DataFrame(y_probas, columns=['prob_hatred'])

	df = pd.concat((df, probabilities),axis=1)
	
	df.to_csv(join(args.PATH_DATA, f'{args.name_data}_preds.csv'))
			