# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(document)

    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        matrix[token.i][token.i] = 1
        for child in token.children:
                matrix[token.i][child.i] = 1
                matrix[child.i][token.i] = 1
    return matrix

def process(filename):

    with open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        lines = fin.readlines()

    idx2graph = {}

    with open(filename+'.graph.new', 'wb') as fout:
        for i in tqdm(range(0, len(lines), 1)):
            text = lines[i].lower().strip()
            adj_matrix = dependency_adj_matrix(text)
            idx2graph[i] = adj_matrix
        pickle.dump(idx2graph, fout)        


if __name__ == '__main__':

#     process('./IAC1/spacy/train.txt')
#     process('./IAC1/spacy/test.txt')
#     # process('./IAC1/spacy/valid.txt')
#     print("IAC1运行完成")
#     process('./IAC2/spacy/train.txt')
#     process('./IAC2/spacy/test.txt')
#     # process('./IAC2/spacy/valid.txt')
#     print("IAC2运行完成")

#     process('./riloff/spacy/train.txt')
#     process('./riloff/spacy/test.txt')
#     print("riloff运行完成")
    
    # process('./Twitter/spacy/train.txt')
    # process('./Twitter/spacy/test.txt')
    # print("Twitter运行完成")
    
    # process('./datasets/Sarcasm_Headlines_Dataset.txt')
    # print("Sarcasm_Headlines_Dataset运行完成")
    
    process('./reddit/spacy/train.txt')
    process('./reddit/spacy/test.txt')
    print("reddit运行完成")
    
    # process('./datasets/ghost/test.txt')
    # process('./datasets/ghost/train.txt')
    # print("ghost运行完成")
    
#     process('./Sarcasm_Headlines_Dataset/spacy/Sarcasm_Headlines_Dataset.txt')
#     print("Sarcasm_Headlines_Dataset运行完成")
    # process('./reddit/spacy/train.txt')
    # print("reddit运行完成")
    
    
    
#     process('./tweet/spacy/train.txt')
#     process('./tweet/spacy/test.txt')
#     # print("tweet运行完成")