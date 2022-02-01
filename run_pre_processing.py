from os import getcwd, mkdir
from os.path import join, isdir

import pandas as pd

PATH_REPO = getcwd()
PATH_DATA = join(PATH_REPO, 'data')

from utils.pre_process import pre_process

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='pre process tweets')
    
    parser.add_argument('--PATH_DATA',   metavar='d', type=str, help='PATH to data folder'  , default=PATH_DATA)
    parser.add_argument('--name', metavar='n', type=str, help='project name')
    parser.add_argument('--zip', metavar='e', type=bool, help='zip', default=False)
    parser.add_argument('--text_var', metavar='t', type=str, help='variable name for text', default='text')
    
    args = parser.parse_args()
    
    if not args.zip:
        df = pd.read_csv(join(args.PATH_DATA, f'{args.name}.csv'), engine='python')
    else:
        df = pd.read_csv(join(args.PATH_DATA, f'{args.name}.zip'), engine='python', compression='zip')
    
    df = pre_process(df, text_var=args.text_var)

    if not args.zip:
        df.to_csv(join(args.PATH_DATA, f'{args.name}_pp.csv'), index=False)
    else:
        df.to_csv(join(args.PATH_DATA, f'{args.name}_pp.zip'), index=False, compression='zip')