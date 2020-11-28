import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_sizes', nargs='*')
args = parser.parse_args()

def parse_to_lists(args):
    args = list()
    return argsmap(ast.literal_eval, args)