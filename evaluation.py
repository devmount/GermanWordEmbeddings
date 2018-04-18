#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to create test-sets for evaluation of word embeddings
# saves logged results in additional file
# 
# @author: Andreas Mueller
# @see: Bachelor Thesis 'Analyse von Wort-Vektoren deutscher Textkorpora'
#
# Contributors:
#  Michael Egger <michael.egger@tsn.at>
#
# @example: python evaluation.py test.model -cu

import gensim
import random
import argparse
import logging
from pathlib import Path

# configuration
parser = argparse.ArgumentParser(description='Script for creating testsets and evaluating word vector models')
parser.add_argument('model', type=str, help='source file with trained model')
parser.add_argument('-c', '--create', action='store_true', help='if set, create testsets before evaluating')
parser.add_argument('-u', '--umlauts', action='store_true', help='if set, create additional testsets with transformed umlauts and use them instead')
parser.add_argument('-t', '--topn', type=int, default=10, help='check the top n result (correct answer under top n answeres)')

args = parser.parse_args()
TARGET_SYN = 'data/syntactic.questions'
TARGET_SEM_OP = 'data/semantic_op.questions'
TARGET_SEM_BM = 'data/semantic_bm.questions'
TARGET_SEM_DF = 'data/semantic_df.questions'
SRC_NOUNS = 'src/nouns.txt'
SRC_ADJECTIVES = 'src/adjectives.txt'
SRC_VERBS = 'src/verbs.txt'
SRC_BESTMATCH = 'src/bestmatch.txt'
SRC_DOESNTFIT = 'src/doesntfit.txt'
SRC_OPPOSITE = 'src/opposite.txt'
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
    ('verbs (past)', '3PV/3SV', SRC_VERBS, 4, 3)
]
logging.basicConfig(filename=args.model.strip() + '.result', format='%(asctime)s : %(message)s', level=logging.INFO)

consoleHandler = logging.StreamHandler()
logging.getLogger().addHandler(consoleHandler)


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


def create_syntactic_testset():
    """
    Creates syntactic test set and writes it into a file.

    :return: None
    """
    if args.umlauts:
        u = open(TARGET_SYN + '.nouml', 'w')
    with open(TARGET_SYN, 'w') as t:
        for label, short, src, index1, index2 in PATTERN_SYN:
            t.write(': {}: {}\n'.format(label, short))
            if args.umlauts:
                u.write(': {}: {}\n'.format(label, short))
            for q in create_questions(src, index1, index2):
                t.write(q + '\n')
                if args.umlauts:
                    u.write(replace_umlauts(q) + '\n')
            logging.info('created pattern ' + short)


def create_semantic_testset():
    """
    Creates semantic test set and writes it into a file.

    :return: None
    """
    # opposite
    with open(TARGET_SEM_OP, 'w') as t:
        for q in create_questions(SRC_OPPOSITE, combinate=10):
            t.write(q + '\n')
            if args.umlauts:
                with open(TARGET_SEM_OP + '.nouml', 'w') as u:
                    u.write(replace_umlauts(q) + '\n')
        logging.info('created opposite questions')

    # best match
    with open(TARGET_SEM_BM, 'w') as t:
        groups = open(SRC_BESTMATCH).read().split(':')
        groups.pop(0)  # remove first empty group
        for group in groups:
            questions = group.splitlines()
            _ = questions.pop(0)
            while questions:
                for i in range(1, len(questions)):
                    question = questions[0].split('-') + questions[i].split('-')
                    t.write(' '.join(question) + '\n')
                    if args.umlauts:
                        with open(TARGET_SEM_BM + '.nouml', 'w') as u:
                            u.write(replace_umlauts(' '.join(question)) + '\n')
                questions.pop(0)
        logging.info('created best-match questions')

    # doesn't fit
    with open(TARGET_SEM_DF, 'w') as t:
        for line in open(SRC_DOESNTFIT):
            words = line.split()
            for wrongword in words[-1].split('-'):
                question = ' '.join(words[:3] + [wrongword])
                t.write(question + '\n')
                if args.umlauts:
                    with open(TARGET_SEM_DF + '.nouml', 'w') as u:
                        u.write(replace_umlauts(question) + '\n')
        logging.info('created doesn\'t-fit questions')


def create_questions(src, index1=0, index2=1, combinate=5):
    """
    Creates single questions from given source.

    :param src: source file to load words from
    :param index1: index of first word in a line to focus on
    :param index2: index of second word in a line to focus on
    :param combinate: combinate number of combinations with random other lines
    :return: list of question words
    """
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


def test_most_similar(model, src, label='most similar', topn=10):
    """
    Tests given model to most similar word.

    :param model: model to test
    :param src: source file to load words from
    :param label: label to print current test case
    :param topn: number of top matches
    :return:
    """
    num_lines = sum(1 for _ in open(src))
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
        if all(x in model.index2word for x in words):
            num_questions += 1
            best_matches = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=topn)
            # best match
            if words[3] in best_matches[0]:
                num_right += 1
            # topn match
            for match in best_matches[:topn]:
                if words[3] in match:
                    num_topn += 1
                    break
    # calculate result
    correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions > 0 else 0.0
    topn_matches = round(num_topn/float(num_questions)*100, 1) if num_questions > 0 else 0.0
    coverage = round(num_questions/float(num_lines)*100, 1) if num_lines > 0 else 0.0
    # log result
    logging.info(label + ' correct:  {0}% ({1}/{2})'.format(correct_matches, num_right, num_questions))
    logging.info(label + ' top {0}:   {1}% ({2}/{3})'.format(topn, topn_matches, num_topn, num_questions))
    logging.info(label + ' coverage: {0}% ({1}/{2})'.format(coverage, num_questions, num_lines))


def test_most_similar_groups(model, src, topn=10):
    """
    Tests given model to most similar word.

    :param model: model to test
    :param src: source file to load words from
    :param topn: number of top matches
    :return: None
    """
    num_lines = 0
    num_questions = 0
    num_right = 0
    num_topn = 0
    # test each group
    with open(src) as groups_fp:
        groups = groups_fp.read().split('\n: ')
        for group in groups:
            questions = group.splitlines()
            label = questions.pop(0)
            label = label[2:] if label.startswith(': ') else label  # handle first group
            num_group_lines = len(questions)
            num_group_questions = 0
            num_group_right = 0
            num_group_topn = 0
            # test each question of current group
            for question in questions:
                words = question.split()
                # check if all words exist in vocabulary
                if all(x in model.index2word for x in words):
                    num_group_questions += 1
                    best_matches = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=topn)
                    # best match
                    if words[3] in best_matches[0]:
                        num_group_right += 1
                    # topn match
                    for match in best_matches[:topn]:
                        if words[3] in match:
                            num_group_topn += 1
                            break
            # calculate result
            correct_group_matches = round(num_group_right/float(num_group_questions)*100, 1) if num_group_questions > 0 else 0.0
            topn_group_matches = round(num_group_topn/float(num_group_questions)*100, 1) if num_group_questions > 0 else 0.0
            group_coverage = round(num_group_questions/float(num_group_lines)*100, 1) if num_group_lines > 0 else 0.0
            # log result
            logging.info(label + ': {0}% ({1}/{2}), {3}% ({4}/{5}), {6}% ({7}/{8})'.format(
                correct_group_matches,
                num_group_right,
                num_group_questions,
                topn_group_matches,
                num_group_topn,
                num_group_questions,
                group_coverage,
                num_group_questions,
                num_group_lines
            ))
            # total numbers
            num_lines += num_group_lines
            num_questions += num_group_questions
            num_right += num_group_right
            num_topn += num_group_topn
        # calculate result
        correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions > 0 else 0.0
        topn_matches = round(num_topn/float(num_questions)*100, 1) if num_questions > 0 else 0.0
        coverage = round(num_questions/float(num_lines)*100, 1) if num_lines > 0 else 0.0
        # log result
        logging.info('total correct:  {0}% ({1}/{2})'.format(correct_matches, num_right, num_questions))
        logging.info('total top {0}:   {1}% ({2}/{3})'.format(topn, topn_matches, num_topn, num_questions))
        logging.info('total coverage: {0}% ({1}/{2})'.format(coverage, num_questions, num_lines))


def test_doesnt_fit(model, src):
    """
    Tests given model to most not fitting word.

    :param model: model to test
    :param src: source file to load words from
    :return:
    """
    num_lines = sum(1 for _ in open(src))
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
        if all(x in model.index2word for x in words):
            num_questions += 1
            if model.doesnt_match(words) == words[3]:
                num_right += 1
    # calculate result
    correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions > 0 else 0.0
    coverage = round(num_questions/float(num_lines)*100, 1) if num_lines > 0 else 0.0
    # log result
    logging.info('doesn\'t fit correct:  {0}% ({1}/{2})'.format(correct_matches, num_right, num_questions))
    logging.info('doesn\'t fit coverage: {0}% ({1}/{2})'.format(coverage, num_questions, num_lines))

if args.create:
    logging.info('> CREATING SYNTACTIC TESTSET')
    create_syntactic_testset()
    logging.info('> CREATING SEMANTIC TESTSET')
    create_semantic_testset()

# get trained model
is_binary = Path(args.model.strip()).suffix == ".bin"
trained_model = gensim.models.KeyedVectors.load_word2vec_format(args.model.strip(), binary=is_binary)
# remove original vectors to free up memory
trained_model.init_sims(replace=True)

# execute evaluation
logging.info('> EVALUATING SYNTACTIC FEATURES')
test_most_similar_groups(trained_model, TARGET_SYN + '.nouml' if args.umlauts else TARGET_SYN, args.topn)
logging.info('> EVALUATING SEMANTIC FEATURES')
test_most_similar(trained_model, TARGET_SEM_OP + '.nouml' if args.umlauts else TARGET_SEM_OP, 'opposite', args.topn)
test_most_similar(trained_model, TARGET_SEM_BM + '.nouml' if args.umlauts else TARGET_SEM_BM, 'best match', args.topn)
test_doesnt_fit(trained_model, TARGET_SEM_DF + '.nouml' if args.umlauts else TARGET_SEM_DF)
