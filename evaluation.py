#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to create test-sets for evaluation of word embeddings
# saves logged results in additional file
# 
# @author: Andreas Mueller
# @see: Bachelor Thesis 'Analyse von Wort-Vektoren deutscher Textkorpora'
# 
# @example: python evaluation.py test.model -cu

import gensim
import random
import argparse
import logging

# configuration
parser = argparse.ArgumentParser(description='Script for creating testsets and evaluating word vector models')
parser.add_argument('model', type=str, help='source file with trained model')
parser.add_argument('-c', '--create', action='store_true', help='if set, create testsets before evaluating')
parser.add_argument('-u', '--umlauts', action='store_true', help='if set, create additional testsets with transformed umlauts and use them instead')
args = parser.parse_args()
TARGET_SYN     = 'data/syntactic.questions'
TARGET_SEM_OP  = 'data/semantic_op.questions'
TARGET_SEM_BM  = 'data/semantic_bm.questions'
TARGET_SEM_DF  = 'data/semantic_df.questions'
SRC_NOUNS      = 'src/nouns.txt'
SRC_ADJECTIVES = 'src/adjectives.txt'
SRC_VERBS      = 'src/verbs.txt'
SRC_BESTMATCH  = 'src/bestmatch.txt'
SRC_DOESNTFIT  = 'src/doesntfit.txt'
SRC_OPPOSITE   = 'src/opposite.txt'
PATTERN_SYN = [
    ('nouns', 'SI/PL', SRC_NOUNS, 0, 1),
    ('nouns', 'PL/SI', SRC_NOUNS, 1, 0),
    ('adjectives', 'GR/KOM', SRC_ADJECTIVES, 0, 1),
    ('adjectives', 'KOM/GR', SRC_ADJECTIVES, 1, 0),
    ('adjectives', 'GR/SUP', SRC_ADJECTIVES, 0, 2),
    ('adjectives', 'SUP/GR', SRC_ADJECTIVES, 2, 0),
    ('adjectives', 'KOM/SUP', SRC_ADJECTIVES, 1, 2),
    ('adjectives', 'SUP/KOM', SRC_ADJECTIVES, 2, 1),
    ('verbs (pres)', 'INF/1SP', SRC_VERBS, 0, 1),
    ('verbs (pres)', '1SP/INF', SRC_VERBS, 1, 0),
    ('verbs (pres)', 'INF/2PP', SRC_VERBS, 0, 2),
    ('verbs (pres)', '2PP/INF', SRC_VERBS, 2, 0),
    ('verbs (pres)', '1SP/2PP', SRC_VERBS, 1, 2),
    ('verbs (pres)', '2PP/1SP', SRC_VERBS, 2, 1),
    ('verbs (past)', 'INF/3SV', SRC_VERBS, 0, 3),
    ('verbs (past)', '3SV/INF', SRC_VERBS, 3, 0),
    ('verbs (past)', 'INF/3PV', SRC_VERBS, 0, 4),
    ('verbs (past)', '3PV/INF', SRC_VERBS, 4, 0),
    ('verbs (past)', '3SV/3PV', SRC_VERBS, 3, 4),
    ('verbs (past)', '3PV/3SV', SRC_VERBS, 4, 3)]
logging.basicConfig(filename=args.model + '.result', format='%(asctime)s : %(message)s', level=logging.INFO)


# function replace_umlauts
# ... replaces german umlauts and sharp s in given text
# @param string  text
# @return string with replaced umlauts
def replace_umlauts(text):
    res = text.decode('utf8')
    res = res.replace(u'ä', 'ae')
    res = res.replace(u'ö', 'oe')
    res = res.replace(u'ü', 'ue')
    res = res.replace(u'Ä', 'Ae')
    res = res.replace(u'Ö', 'Oe')
    res = res.replace(u'Ü', 'Ue')
    res = res.replace(u'ß', 'ss')
    return res


# function create_syntactic_testset
# ... creates syntactic test set and writes it into a file
# @return void
def create_syntactic_testset():
    if args.umlauts:
        u = open(TARGET_SYN + '.nouml', 'w')
    with open(TARGET_SYN, 'w') as t:
        for label, short, src, index1, index2 in PATTERN_SYN:
            t.write(': ' + label + ': ' + short + '\n')
            if args.umlauts:
                u.write(': ' + label + ': ' + short + '\n')
            for q in create_questions(src, index1, index2):
                t.write(q + '\n')
                if args.umlauts:
                    u.write(replace_umlauts(q) + '\n')
            logging.info('created pattern ' + short)


# function create_semantic_testset
# ... creates semantic test set and writes it into a file
# @return void
def create_semantic_testset():
    # opposite
    if args.umlauts:
        u = open(TARGET_SEM_OP + '.nouml', 'w')
        u.write(': opposite\n')
    with open(TARGET_SEM_OP, 'w') as t:
        t.write(': opposite\n')
        for q in create_questions(SRC_OPPOSITE, combinate=10):
            t.write(q + '\n')
            if args.umlauts:
                u.write(replace_umlauts(q) + '\n')
        logging.info('created opposite questions')
    # best match
    if args.umlauts:
        u = open(TARGET_SEM_BM + '.nouml', 'w')
    with open(TARGET_SEM_BM, 'w') as t:
        groups = open(SRC_BESTMATCH).read().split(':')
        groups.pop(0) # remove first empty group
        for group in groups:
            questions = group.splitlines()
            label = questions.pop(0)
            while questions:
                for i in range(1,len(questions)):
                    question = questions[0].split('-') + questions[i].split('-')
                    t.write(' '.join(question) + '\n')
                    if args.umlauts:
                        u.write(replace_umlauts(' '.join(question)) + '\n')
                questions.pop(0)
        logging.info('created best-match questions')
    # doesn't fit
    if args.umlauts:
        u = open(TARGET_SEM_DF + '.nouml', 'w')
    with open(TARGET_SEM_DF, 'w') as t:
        for line in open(SRC_DOESNTFIT):
            words = line.split()
            for wrongword in words[-1].split('-'):
                question = ' '.join(words[0:2] + [wrongword])
                t.write(question + '\n')
                if args.umlauts:
                    u.write(replace_umlauts(question) + '\n')
        logging.info('created doesn\'t-fit questions')


# function create_questions
# ... creates single questions from given source
# @param string  src    source file to load words from
# @param integer index2    index of first word in a line to focus on
# @param integer index2    index of second word in a line to focus on
# @param integer combinate number of combinations with random other lines
# @return list of question words
def create_questions(src, index1=0, index2=1, combinate=5):
    # get source content
    with open(src) as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    questions = []

    for line in content:
        for i in range(0, combinate):
            # get current word pair
            question = list(line.split('-')[i] for i in [index1, index2])
            # get random word pair that is not the current
            random_line = random.choice(list(set(content) - {line}))
            random_word = list(random_line.split('-')[i] for i in [index1, index2])
            # merge both word pairs to one question
            question.extend(random_word)
            questions.append(' '.join(question))
    return questions


# function test_bestmatch
# ... tests given model to best matching word
# @param word2vec model to test
# @param string   src   source file to load words from
# @param integer  topn  source file to load words from
def test_bestmatch(model, src, topn=10):
    num_lines = sum(1 for line in open(src))
    num_questions = 0
    num_right = 0
    num_topn = 0
    # get questions
    with open(src) as f:
        questions = f.readlines()
        questions = [x.strip() for x in questions]
    # test each question
    for question in questions:
        words = question.split()
        # check if all words exist in vocabulary
        if set(model.index2word).issubset(words):
            num_questions += 1
            bestmatches = model.most_similar(positive=[words[1], words[2]], negative=[words[0]])
            for topnmatches in bestmatches[:topn]:
                # best match
                if words[3] in topnmatches[0]:
                    num_right += 1
                # topn match
                if words[3] in topnmatches:
                    num_topn += 1
    # calculate result
    correct_matches = num_right/num_questions*100 if num_questions>0 else 0.0
    topn_matches = num_topn/num_questions*100 if num_questions>0 else 0.0
    coverage = num_questions/num_lines*100 if num_lines>0 else 0.0
    # log result
    logging.info('best match correct:   {0}% ({1}/{2})'.format(str(correct_matches), str(num_right), str(num_questions)))
    logging.info('best match top {0}:    {1}% ({2}/{3})'.format(str(topn), str(topn_matches), str(num_topn), str(num_questions)))
    logging.info('best match coverage:  {0}% ({1}/{2})'.format(str(coverage), str(num_questions), str(num_lines)))
        

# function test_doesntfit
# ... tests given model to best not fitting word
# @param word2vec model to test
# @param string   src   source file to load words from
def test_doesntfit(model, src):
    num_lines = sum(1 for line in open(src))
    num_questions = 0
    num_right = 0
    # get questions
    with open(src) as f:
        questions = f.readlines()
        questions = [x.strip() for x in questions]
    # test each question
    for question in questions:
        words = question.split()
        # check if all words exist in vocabulary
        if set(model.index2word).issubset(words):
            num_questions += 1
            if model.doesnt_match(words) == words[3]:
                num_right += 1
    # calculate result
    correct_matches = num_right/num_questions*100 if num_questions>0 else 0.0
    coverage = num_questions/num_lines*100 if num_lines>0 else 0.0
    # log result
    logging.info('doesn\'t fit correct:  {0}% ({1}/{2})'.format(str(correct_matches), str(num_right), str(num_questions)))
    logging.info('doesn\'t fit coverage: {0}% ({1}/{2})'.format(str(coverage), str(num_questions), str(num_lines)))
                

if args.create:
    logging.info('> CREATING SYNTACTIC TESTSET')
    create_syntactic_testset()
    logging.info('> CREATING SEMANTIC TESTSET')
    create_semantic_testset()

# get trained model
model = gensim.models.Word2Vec.load_word2vec_format(args.model, binary=True)
# execute evaluation
logging.info('> EVALUATING SYNTACTIC FEATURES')
model.accuracy(TARGET_SYN + '.nouml' if args.umlauts else TARGET_SYN, restrict_vocab=50000)
logging.info('> EVALUATING SEMANTIC FEATURES')
model.accuracy(TARGET_SEM_OP + '.nouml' if args.umlauts else TARGET_SEM_OP, restrict_vocab=50000)
test_bestmatch(model, TARGET_SEM_BM + '.nouml' if args.umlauts else TARGET_SEM_BM)
test_doesntfit(model, TARGET_SEM_DF + '.nouml' if args.umlauts else TARGET_SEM_DF)
