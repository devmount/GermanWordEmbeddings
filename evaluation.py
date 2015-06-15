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
parser.add_argument('-t', '--topn', type=int, default=10, help='check the top n result (correct answer under top n answeres)')
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
logging.basicConfig(filename=args.model.strip() + '.top' + args.topn + '.result', format='%(asctime)s : %(message)s', level=logging.INFO)


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
    with open(TARGET_SEM_OP, 'w') as t:
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
                question = ' '.join(words[:3] + [wrongword])
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


# function test_mostsimilar
# ... tests given model to most similar word
# @param word2vec model to test
# @param string   src   source file to load words from
# @param string   label to print current test case
# @param integer  topn  number of top matches
def test_mostsimilar(model, src, label='most similar', topn=10):
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
        words = question.decode('utf-8').split()
        # check if all words exist in vocabulary
        if all(x in model.index2word for x in words):
            num_questions += 1
            bestmatches = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=topn)
            # best match
            if words[3] in bestmatches[0]:
                num_right += 1
            # topn match
            for topmatches in bestmatches[:topn]:
                if words[3] in topmatches:
                    num_topn += 1
                    break
    # calculate result
    correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions>0 else 0.0
    topn_matches = round(num_topn/float(num_questions)*100, 1) if num_questions>0 else 0.0
    coverage = round(num_questions/float(num_lines)*100, 1) if num_lines>0 else 0.0
    # log result
    logging.info(label + ' correct:  {0}% ({1}/{2})'.format(str(correct_matches), str(num_right), str(num_questions)))
    logging.info(label + ' top {0}:   {1}% ({2}/{3})'.format(str(topn), str(topn_matches), str(num_topn), str(num_questions)))
    logging.info(label + ' coverage: {0}% ({1}/{2})'.format(str(coverage), str(num_questions), str(num_lines)))


# function test_mostsimilar
# ... tests given model to most similar word
# @param word2vec model to test
# @param string   src   source file to load words from
# @param integer  topn  number of top matches
def test_mostsimilar_groups(model, src, topn=10):
    num_lines = 0
    num_questions = 0
    num_right = 0
    num_topn = 0
    # test each group
    groups = open(src).read().split('\n: ')
    for group in groups:
        questions = group.splitlines()
        label = questions.pop(0)
        label = label[2:] if label.startswith(': ') else label # handle first group
        num_group_lines = len(questions)
        num_group_questions = 0
        num_group_right = 0
        num_group_topn = 0
        # test each question of current group
        for question in questions:
            words = question.decode('utf-8').split()
            # check if all words exist in vocabulary
            if all(x in model.index2word for x in words):
                num_group_questions += 1
                bestmatches = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=topn)
                # best match
                if words[3] in bestmatches[0]:
                    num_group_right += 1
                # topn match
                for topmatches in bestmatches[:topn]:
                    if words[3] in topmatches:
                        num_group_topn += 1
                        break
        # calculate result
        correct_group_matches = round(num_group_right/float(num_group_questions)*100, 1) if num_group_questions>0 else 0.0
        topn_group_matches = round(num_group_topn/float(num_group_questions)*100, 1) if num_group_questions>0 else 0.0
        group_coverage = round(num_group_questions/float(num_group_lines)*100, 1) if num_group_lines>0 else 0.0
        # log result
        logging.info(label + ': {0}% ({1}/{2}), {3}% ({4}/{5}), {6}% ({7}/{8})'.format(str(correct_group_matches), str(num_group_right), str(num_group_questions), str(topn_group_matches), str(num_group_topn), str(num_group_questions), str(group_coverage), str(num_group_questions), str(num_group_lines)))
        # total numbers
        num_lines += num_group_lines
        num_questions += num_group_questions
        num_right += num_group_right
        num_topn += num_group_topn
    # calculate result
    correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions>0 else 0.0
    topn_matches = round(num_topn/float(num_questions)*100, 1) if num_questions>0 else 0.0
    coverage = round(num_questions/float(num_lines)*100, 1) if num_lines>0 else 0.0
    # log result
    logging.info('total correct:  {0}% ({1}/{2})'.format(str(correct_matches), str(num_right), str(num_questions)))
    logging.info('total top {0}:   {1}% ({2}/{3})'.format(str(topn), str(topn_matches), str(num_topn), str(num_questions)))
    logging.info('total coverage: {0}% ({1}/{2})'.format(str(coverage), str(num_questions), str(num_lines)))


# function test_doesntfit
# ... tests given model to most not fitting word
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
        words = question.decode('utf-8').split()
        # check if all words exist in vocabulary
        if all(x in model.index2word for x in words):
            num_questions += 1
            if model.doesnt_match(words) == words[3]:
                num_right += 1
    # calculate result
    correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions>0 else 0.0
    coverage = round(num_questions/float(num_lines)*100, 1) if num_lines>0 else 0.0
    # log result
    logging.info('doesn\'t fit correct:  {0}% ({1}/{2})'.format(str(correct_matches), str(num_right), str(num_questions)))
    logging.info('doesn\'t fit coverage: {0}% ({1}/{2})'.format(str(coverage), str(num_questions), str(num_lines)))
                
if args.create:
    logging.info('> CREATING SYNTACTIC TESTSET')
    create_syntactic_testset()
    logging.info('> CREATING SEMANTIC TESTSET')
    create_semantic_testset()

# get trained model
model = gensim.models.Word2Vec.load_word2vec_format(args.model.strip(), binary=True)
# execute evaluation
logging.info('> EVALUATING SYNTACTIC FEATURES')
test_mostsimilar_groups(model, TARGET_SYN + '.nouml' if args.umlauts else TARGET_SYN, args.topn)
# logging.info('> EVALUATING SEMANTIC FEATURES')
# test_mostsimilar(model, TARGET_SEM_OP + '.nouml' if args.umlauts else TARGET_SEM_OP, 'opposite', args.topn)
# test_mostsimilar(model, TARGET_SEM_BM + '.nouml' if args.umlauts else TARGET_SEM_BM, 'best match', args.topn)
# test_doesntfit(model, TARGET_SEM_DF + '.nouml' if args.umlauts else TARGET_SEM_DF)
