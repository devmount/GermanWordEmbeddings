#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to create vocabulary of given model
# 
# @author: Andreas Mueller
# @see: Bachelor Thesis 'Analyse von Wort-Vektoren deutscher Textkorpora'
# 
# @example: python vocabulary.py test.model test.model.vocab

import gensim
import argparse

# configuration
parser = argparse.ArgumentParser(description='Script for computing vocabulary of given corpus')
parser.add_argument('model', type=str, help='source file with trained model')
parser.add_argument('target', type=str, help='target file name to store vocabulary in')
args = parser.parse_args()

# load model
model = gensim.models.Word2Vec.load_word2vec_format(args.model, binary=True)

# build vocab
vocab = []
for word,obj in model.vocab.iteritems():
    vocab.append([word,obj.count])

# save vocab
with open(args.target, 'w') as f:
    for word,count in sorted(vocab, key=lambda x: x[1], reverse=True):
        f.write(str(count) + ' ' + word.encode('utf8') + '\n')
