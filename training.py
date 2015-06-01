#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to train word embeddings with word2vec
# 
# @author: Andreas Mueller
# @see: Bachelor Thesis 'Analyse von Wort-Vektoren deutscher Textkorpora'
# 
# @example: python training.py corpus_dir/ test.model -s 300 -w 10

import gensim
import logging
import os
import argparse

# configuration
parser = argparse.ArgumentParser(description='Script for training word vector models using public corpora')
parser.add_argument('corpora', type=str, help='source folder with preprocessed corpora (one sentence plain text per line in each file)')
parser.add_argument('target', type=str, help='target file name to store model in')
parser.add_argument('-s', '--size', type=int, default=100, help='dimension of word vectors')
parser.add_argument('-w', '--window', type=int, default=5, help='size of the sliding window')
parser.add_argument('-m', '--mincount', type=int, default=5, help='minimum number of occurences of a word to be considered')
parser.add_argument('-c', '--workers', type=int, default=4, help='number of worker threads to train the model')
parser.add_argument('-g', '--sg', type=int, default=1, help='training algorithm: Skip-Gram (1), otherwise CBOW (0)')
parser.add_argument('-i', '--hs', type=int, default=1, help='use of hierachical sampling for training')
parser.add_argument('-n', '--negative', type=int, default=0, help='use of negative sampling for training (usually between 5-20)')
parser.add_argument('-o', '--cbowmean', type=int, default=0, help='for CBOW training algorithm: use sum (0) or mean (1) to merge context vectors')
args = parser.parse_args()
logging.basicConfig(filename=args.target.strip() + '.result', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# get corpus sentences
class CorpusSentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = CorpusSentences(args.corpora)

# train the model
model = gensim.models.Word2Vec(
    sentences,
    size=args.size,
    window=args.window,
    min_count=args.mincount,
    workers=args.workers,
    sg=args.sg,
    hs=args.hs,
    negative=args.negative,
    cbow_mean=args.cbowmean
)

# store model
model.save_word2vec_format(args.target, binary=True)
