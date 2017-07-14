#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to preprocess corpora for training
#
# @author: Andreas Mueller
# @see: Bachelor Thesis 'Analyse von Wort-Vektoren deutscher Textkorpora'
#
# @example: python preprocessing.py test.raw test.corpus -psub

import gensim
import nltk.data
from nltk.corpus import stopwords
import argparse
import os
import re
import logging
import sys
import multiprocessing as mp

# configuration
parser = argparse.ArgumentParser(description='Script for preprocessing public corpora')
parser.add_argument('raw', type=str, help='source file with raw data for corpus creation')
parser.add_argument('target', type=str, help='target file name to store corpus in')
parser.add_argument('-p', '--punctuation', action='store_true', help='remove punctuation tokens')
parser.add_argument('-s', '--stopwords', action='store_true', help='remove stop word tokens')
parser.add_argument(
    '-u', '--umlauts', action='store_true', help='replace german umlauts with their respective digraphs'
)
parser.add_argument('-b', '--bigram', action='store_true', help='detect and process common bigram phrases')
parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='thread count')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for multiprocessing')
args = parser.parse_args()
logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
punctuation_tokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '\'', '[', ']',
                      '{', '}', '?', '!', '-', '–', '+', '*', '--', '\'\'', '``']
punctuation = '?.!/;:()&+'


def replace_umlauts(text):
    """
    Replaces german umlauts and sharp s in given text.

    :param text: text as str
    :return: manipulated text as str
    """
    res = text
    res = res.replace('ä', 'ae')
    res = res.replace('ö', 'oe')
    res = res.replace('ü', 'ue')
    res = res.replace('Ä', 'Ae')
    res = res.replace('Ö', 'Oe')
    res = res.replace('Ü', 'Ue')
    res = res.replace('ß', 'ss')
    return res


def process_line(line):
    """
    Pre processes the given line.

    :param line: line as str
    :return: preprocessed sentence
    """
    # detect sentences
    sentences = sentence_detector.tokenize(line)
    # process each sentence
    for sentence in sentences:
        # replace umlauts
        if args.umlauts:
            sentence = replace_umlauts(sentence)
        # get word tokens
        words = nltk.word_tokenize(sentence)
        # filter punctuation and stopwords
        if args.punctuation:
            words = [x for x in words if x not in punctuation_tokens]
            words = [re.sub('[{}]'.format(punctuation), '', x) for x in words]
        if args.stopwords:
            words = [x for x in words if x not in stop_words]
        # write one sentence per line in output file, if sentence has more than 1 word
        if len(words) > 1:
            return '{}\n'.format(' '.join(words))

# get stopwords
if not args.umlauts:
    stop_words = stopwords.words('german')
else:
    stop_words = [replace_umlauts(token) for token in stopwords.words('german')]

if not os.path.exists(os.path.dirname(args.target)):
    os.makedirs(os.path.dirname(args.target))
with open(args.raw, 'r') as infile:
    # start pre processing with multiple threads
    pool = mp.Pool(args.threads)
    values = pool.imap(process_line, infile, chunksize=args.batch_size)
    with open(args.target, 'w') as outfile:
        for i, s in enumerate(values):
            if i and i % 25000 == 0:
                logging.info('processed {} sentences'.format(i))
                outfile.flush()
            if s:
                outfile.write(s)
        logging.info('preprocessing of {} sentences finished!'.format(i))


# get corpus sentences
class CorpusSentences:
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()

if args.bigram:
    logging.info('train bigram phrase detector')
    bigram = gensim.models.Phrases(CorpusSentences(args.target))
    logging.info('transform corpus to bigram phrases')
    with open('{}.bigram'.format(args.target), 'w') as outfile:
        for tokens in bigram[CorpusSentences(args.target)]:
            outfile.write('{}\n'.format(' '.join(tokens)))
