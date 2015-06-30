#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to visualize word embeddings of given model with PCA dimensionality reduction
# creates image with matplotlib
# 
# @author: Andreas Mueller
# @see: Bachelor Thesis 'Analyse von Wort-Vektoren deutscher Textkorpora'
# 
# @example: python visualize.py test.model -cu

import gensim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# configuration
currency = ["Schweiz","Franken","Deutschland","Euro","Grossbritannien","britische_Pfund","Japan","Yen","Russland","Rubel","USA","US-Dollar","Kroatien","Kuna"] #"Pakistan","Rupien","China","Yuan",
capital  = ["Athen","Griechenland","Berlin","Deutschland","Ankara","Tuerkei","Bern","Schweiz","Hanoi","Vietnam","Lissabon","Portugal","Moskau","Russland","Stockholm","Schweden","Tokio","Japan","Washington","USA"]
language = ["Deutschland","Deutsch","USA","Englisch","Frankreich","Franzoesisch","Griechenland","Griechisch","Norwegen","Norwegisch","Schweden","Schwedisch","Polen","Polnisch","Ungarn","Ungarisch"]
# matches = model.most_similar(positive=["Frau"], negative=[], topn=30)
# words = [match[0] for match in matches]

# function draw_words
# ... reduces dimensionality of vectors of given words either with PCA or with t-SNE and draws the words into a diagram
# @param word2vec model     to visualize vectors from
# @param list     words     list of word strings to visualize
# @param bool     alternate use different color and align for every second word
# @param bool     pca       use PCA (True) or t-SNE (False) to reduce dimensionality 
def draw_words(model, words, alternate=False, pca=True, title=''):
    # get vectors for given words from model
    vectors = [model[word] for word in words]

    if pca:
        pca = PCA(n_components=2, whiten=True)
        vectors2d = pca.fit(vectors).transform(vectors)
    else:
        tsne = TSNE(n_components=2, random_state=0)
        vectors2d = tsne.fit_transform(vectors)

    # draw image
    plt.figure()
    plt.axis([-2.2,2.2,-2.2,2.2])

    first = True # color alternation to divide given groups
    for point, word in zip(vectors2d , words):
        # plot points
        plt.scatter(
            point[0],
            point[1],
            c='r'
        )
        # plot word annotations
        plt.annotate(
            word, 
            xy = (point[0], point[1]),
            xytext = (-7, -6) if first else (7, -6),
            textcoords = 'offset points',
            ha = 'right' if first else 'left',
            va = 'bottom',
            size = "x-large"
        )
        first = not first if alternate else first

    # draw arrows
    for i in xrange(0, len(words)-1, 2):
        a = vectors2d[i][0] + 0.04
        b = vectors2d[i][1]
        c = vectors2d[i+1][0] - 0.04
        d = vectors2d[i+1][1]
        plt.arrow(a, b, c-a, d-b,
            shape='full',
            lw=0.1,
            edgecolor='#bbbbbb',
            facecolor='#bbbbbb',
            length_includes_head=True,
            head_width=0.08,
            width=0.01
        )

    # draw diagram title
    if title:
        plt.title(title)

    plt.show()

# get trained model
model = gensim.models.Word2Vec.load_word2vec_format("model/SG-300-5-NS10-R50.model", binary=True)
# execute evaluation
draw_words(model, currency, True, True, r'$PCA\ Visualisierung:\ W\ddot{a}hrung$')
draw_words(model, capital, True, True, r'$PCA\ Visualisierung:\ Hauptstadt$')
draw_words(model, language, True, True, r'$PCA\ Visualisierung:\ Sprache$')
