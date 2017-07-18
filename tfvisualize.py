#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to visualize trained model with tensorboard
#
# @author: Michael Egger <michael.egger@tsn.at>
#
# @example: python tfvisualize.py test.model

import argparse
import os

import gensim
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


parser = argparse.ArgumentParser(description='Script for visualizing word vector models with tensorboard')
parser.add_argument('model', type=str, help='source file with trained model')
parser.add_argument('-s', '--samples', type=int, default=10000, help='number of samples to project')
parser.add_argument('-p', '--projector', type=str, default='projector', help='target projector path')
parser.add_argument('--prefix', type=str, default='default', help='model prefix for projector files')
args = parser.parse_args()

# create projector folder if it doesnt exist
if not os.path.exists(args.projector):
    os.makedirs(args.projector)

# loading your gensim
model = gensim.models.KeyedVectors.load_word2vec_format(args.model, binary=True)

# project part of vocab with all dimensions
w2v_samples = np.zeros((args.samples, model.vector_size))
with open('{}/{}_metadata.tsv'.format(args.projector, args.prefix), 'w+') as file_metadata:
    for i, word in enumerate(model.wv.index2word[:args.samples]):
        w2v_samples[i] = model[word]
        file_metadata.write(word + '\n')

# define the model without training
sess = tf.InteractiveSession()

with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v_samples, trainable=False, name='{}_embedding'.format(args.prefix))

tf.global_variables_initializer().run()

saver = tf.train.Saver()
writer = tf.summary.FileWriter(args.projector, sess.graph)

# adding into projector
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = '{}_embedding'.format(args.prefix)
embed.metadata_path = './{}_metadata.tsv'.format(args.prefix)

# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(writer, config)

saver.save(sess, '{}/{}_model.ckpt'.format(args.projector, args.prefix), global_step=args.samples)

print('Start tensorboard with: \'tensorboard --logdir=\"projector\"\'\n'
      'and check http://localhost:6006/#embeddings to view your embedding')
