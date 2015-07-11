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

# configuration
parser = argparse.ArgumentParser(description='Script for preprocessing public corpora')
parser.add_argument('raw', type=str, help='source file with raw data for corpus creation')
parser.add_argument('target', type=str, help='target file name to store corpus in')
parser.add_argument('-p', '--punctuation', action='store_true', help='remove punctuation tokens')
parser.add_argument('-s', '--stopwords', action='store_true', help='remove stop word tokens')
parser.add_argument('-u', '--umlauts', action='store_true', help='replace german umlauts with their respective digraphs')
parser.add_argument('-b', '--bigram', action='store_true', help='detect and process common bigram phrases')
args = parser.parse_args()
logging.basicConfig(stream=sys.stdout, ormat='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
punctuation_tokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '\'', '[', ']', '{', '}', '?', '!', '-', u'–', '+', '*', '--', '\'\'', '``']
punctuation = '?.!/;:()&+'


# function replace_umlauts
# ... replaces german umlauts and sharp s in given text
# @param string  text
# @return string with replaced umlauts
def replace_umlauts(text):
    res = text
    res = res.replace(u'ä', 'ae')
    res = res.replace(u'ö', 'oe')
    res = res.replace(u'ü', 'ue')
    res = res.replace(u'Ä', 'Ae')
    res = res.replace(u'Ö', 'Oe')
    res = res.replace(u'Ü', 'Ue')
    res = res.replace(u'ß', 'ss')
    return res

# get stopwords
stop_words = stopwords.words('german') if not args.umlauts else [replace_umlauts(token) for token in stopwords.words('german')]

# start preprocessing
num_sentences = sum(1 for line in open(args.raw))
# if not os.path.exists(os.path.dirname(args.target)):
    # os.makedirs(os.path.dirname(args.target))
output = open(args.target, 'w')
i = 1

logging.info('preprocessing ' + str(num_sentences) + ' sentences')
with open(args.raw, 'r') as infile:
    for line in infile:
        # detect sentences
        sentences = sentence_detector.tokenize(line.decode('utf8'))
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
                words = [re.sub('[' + punctuation + ']', '', x) for x in words]
            if args.stopwords:
                words = [x for x in words if x not in stop_words]
            # write one sentence per line in output file, if sentence has more than 1 word
            if len(words)>1:
                output.write(' '.join(words).encode('utf8') + '\n')
        # logging.info('preprocessing sentence ' + str(i) + ' of ' + str(num_sentences))
        i += 1
logging.info('preprocessing of ' + str(num_sentences) + ' sentences finished!')

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
    output = open(args.target + '.bigram', 'w')
    for tokens in bigram[CorpusSentences(args.target)]:
        output.write(' '.join(tokens).encode('utf8') + '\n')
