# [GermanWordEmbeddings](https://devmount.github.io/GermanWordEmbeddings/)

[![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](./LICENSE)
<a href="https://devmount.github.io/GermanWordEmbeddings/#download" title="downloads of german.model">
![downloads](https://img.shields.io/badge/downloads-5k-blue.svg?style=flat-square)
</a>

There has been a lot of research about the training of word embeddings on English corpora. This toolkit applies deep learning via [gensims's word2vec](https://radimrehurek.com/gensim/models/word2vec.html) on German corpora to train and evaluate German language models. An overview about the project, evaluation results and [download links](https://devmount.github.io/GermanWordEmbeddings/#download) can be found on the [project's website](https://devmount.github.io/GermanWordEmbeddings/) or directly in this repository.

This project is released under the [MIT license](MIT.md).

1. [Get started](#getstarted)
2. [Obtaining corpora](#obtention)
3. [Preprocessing](#preprocessing)
4. [Training models](#training)
5. [Vocabulary](#vocabulary)
6. [Evaluation](#evaluation)
7. [Download](#download)

## Get started <a name="getstarted"></a>

Make sure you have **Python 3** installed, as well as the following libraries:

```shell
pip install gensim nltk matplotlib numpy scipy scikit-learn
```

Now you can download [`word2vec_german.sh`](./word2vec_german.sh) and execute it in your shell to automatically download this toolkit and the corresponding corpus files and do the model training and evaluation. Be aware that this could take a **huge amount of time**!

You can also clone this repository and use my already [trained model](http://cloud.devmount.de/d2bc5672c523b086) to play around with the evaluation and visualization.

If you just want to see how the different Python scripts work, have a look into the [code directory](./code) to see Jupyter Notebook script output examples.

## Obtaining corpora <a name="obtention"></a>

There are multiple possibilities for obtaining huge German corpora that are publicly available and free to use:

### German Wikipedia

```shell
wget https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2
```

### Statistical Machine Translation

Shuffled German news of the years 2007 to 2013:

```shell
for i in 2007 2008 2009 2010 2011 2012 2013; do
  wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.$i.de.shuffled.gz
done
```

Models trained with this toolkit are based on the German Wikipedia and German news of 2013.

## Preprocessing <a name="preprocessing"></a>

This Tool preprocesses the raw wikipedia XML corpus with the WikipediaExtractor (a Python Script from Giuseppe Attardi to filter a Wikipedia XML Dump, licensed under GPLv3) and some shell instructions to filter all XML tags and quotations:

```shell
wget http://medialab.di.unipi.it/Project/SemaWiki/Tools/WikiExtractor.py
python WikiExtractor.py -c -b 25M -o extracted dewiki-latest-pages-articles.xml.bz2
find extracted -name '*bz2' \! -exec bzip2 -k -c -d {} \; > dewiki.xml
sed -i 's/<[^>]*>//g' dewiki.xml
sed -i 's|["'\''„“‚‘]||g' dewiki.xml
rm -rf extracted
```

The German news already contain one sentence per line and don't have any XML syntax overhead. Only quotation should to be removed:

```shell
for i in 2007 2008 2009 2010 2011 2012 2013; do
  gzip -d news.$i.de.shuffled.gz
  sed -i 's|["'\''„“‚‘]||g' news.$i.de.shuffled
done
```

Afterwards, the [`preprocessing.py`](preprocessing.py) script can be called on these corpus files with the following options:

flag                  | default | description
--------------------- | ------- | ---------------------------------------------
-h, --help            | -       | show a help message and exit
-p, --punctuation     | False   | filter punctuation tokens
-s, --stopwords       | False   | filter stop word tokens
-u, --umlauts         | False   | replace german umlauts with their respective digraphs
-b, --bigram          | False   | detect and process common bigram phrases
-t [ ], --threads [ ] | NUMBER_OF_PROCESSORS | number of worker threads
--batch_size [ ]      | 32      | batch size for sentence processing

Example usage:

```shell
python preprocessing.py dewiki.xml corpus/dewiki.corpus -psub
for file in *.shuffled; do python preprocessing.py $file corpus/$file.corpus -psub; done
```

## Training models <a name="training"></a>

Models are trained with the help of the [`training.py`](training.py) script with the following options:

flag                   | default | description
---------------------- | ------- | -----------------------------------------------------
-h, --help             | -       | show this help message and exit
-s [ ], --size [ ]     | 100     | dimension of word vectors
-w [ ], --window [ ]   | 5       | size of the sliding window
-m [ ], --mincount [ ] | 5       | minimum number of occurences of a word to be considered
-t [ ], --threads [ ]  | NUMBER_OF_PROCESSORS | number of worker threads to train the model
-g [ ], --sg [ ]       | 1       | training algorithm: Skip-Gram (1), otherwise CBOW (0)
-i [ ], --hs [ ]       | 1       | use of hierachical sampling for training
-n [ ], --negative [ ] | 0       | use of negative sampling for training (usually between 5-20)
-o [ ], --cbowmean [ ] | 0       | for CBOW training algorithm: use sum (0) or mean (1) to merge context vectors

Example usage:

```shell
python training.py corpus/ my.model -s 200 -w 5
```

Mind that the first parameter is a directory and that every contained file will be taken as a corpus file for training.

If the time needed to train the model should be measured and stored into the results file, this would be a possible command:

```shell
{ time python training.py corpus/ my.model -s 200 -w 5; } 2>> my.model.result
```

## Vocabulary <a name="vocabulary"></a>

To compute the vocabulary of a given corpus, the [`vocabulary.py`](vocabulary.py) script can be used:

```shell
python vocabulary.py my.model my.model.vocab
```

## Evaluation <a name="evaluation"></a>

To create test sets and evaluate trained models, the [`evaluation.py`](evaluation.py) script can be used. It's possible to evaluate both syntactic and semantic features of a trained model. For a successful creation of testsets, the following source files should be created before starting the script (see the configuration part in the script for more information).

### Syntactic test set

With the syntactic test, features like singular, plural, 3rd person, past tense, comparative or superlative can be evaluated. Therefore there are 3 source files: adjectives, nouns and verbs. Every file contains a unique word with its conjugations per line, divided bei a dash. These combination patterns can be entered in the `PATTERN_SYN` constant in the script configuration. The script now combinates each word with 5 random other words according to the given pattern, to create appropriate analogy questions. Once the data file with the questions is created, it can be evaluated. Normally the evaluation can be done by [gensim's word2vec accuracy function](http://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.accuracy), but to get a more specific evaluation result (correct matches, top n matches and coverage), this project uses it's own accuracy functions (`test_mostsimilar_groups()` and `test_mostsimilar()` in [`evaluation.py`](evaluation.py)).

The given source files of this project contains 100 unique nouns with 2 patterns, 100 unique adjectives with 6 patterns and 100 unique verbs with 12 patterns, resulting in 10k analogy questions. Here are some examples for possible source files:

#### adjectives.txt

Possible pattern: `basic-comparative-superlative`

Example content:

```plain
gut-besser-beste
laut-lauter-lauteste
```

See [src/adjectives.txt](src/adjectives.txt)

#### nouns.txt

Possible pattern: `singular-plural`

Example content:

```plain
Bild-Bilder
Name-Namen
```

See [src/nouns.txt](src/nouns.txt)

#### verbs.txt

Possible pattern: `basic-1stPersonSingularPresent-2ndPersonPluralPresent-3rdPersonSingularPast-3rdPersonPluralPast`

Example content:

```plain
finden-finde-findet-fand-fanden
suchen-suche-sucht-suchte-suchten
```

See [src/verbs.txt](src/verbs.txt)

### Semantic test set

With the semantic test, features concering word meanings can be evaluated. Therefore there are 3 source files: opposite, best match and doesn't match. The given source files result in a total of 950 semantic questions.

#### opposite.txt

This file contains opposite words, following the pattern of `oneword-oppositeword` per line, to evaluate the models' ability to find opposites.. The script combinates each pair with 10 random other pairs, to build analogy questions. The given opposite source file of this project includes 30 unique pairs, resulting in 300 analogy questions.

Example content:

```plain
Sommer-Winter
Tag-Nacht
```

See [src/opposite.txt](src/opposite.txt)

#### bestmatch.txt

This file contains groups of content similar word pairs, to evaluate the models ability to find thematic relevant analogies. The script combines each pair with all other pairs of the same group to build analogy questions. The given bestmatch source file of this project includes 7 groups with a total of 77 unique pairs, resulting in 540 analogy questions.

Example content:

```plain
: Politik
Elisabeth-Königin
Charles-Prinz
: Technik
Android-Google
iOS-Apple
Windows-Microsoft
```

See [src/bestmatch.txt](src/bestmatch.txt)

#### doesntfit.txt

This file contains 3 words (per line) with similar content divided by space and a set of words that do not fit, divided by dash, like `fittingword1 fittingword2 fittingword3 notfittingword1-notfittingword2-...-notfittingwordn`. This tests the models' ability to find the least fitting word in a set of 4 words. The script combines each matching triple with every not matching word of the list divided by dash, to build doesntfit questions. The available doesntfit source file of this project includes 11 triples, each with 10 words that do not fit, resulting in 110 questions.

Example content:

```plain
Hase Hund Katze Baum-Besitzer-Elefant-Essen-Haus-Mensch-Tier-Tierheim-Wiese-Zoo
August April September Jahr-Monat-Tag-Stunde-Minute-Zeit-Kalender-Woche-Quartal-Uhr
```

See [src/doesntfit.txt](src/doesntfit.txt)

Those options for the script execution are possible:

flag          | description
------------- | -----------------------------------------------------
-h, --help    | show a help message and exit
-c, --create  | if set, create testsets before evaluating
-u, --umlauts | if set, create additional testsets with transformed umlauts and/or use them instead

Example usage:

```shell
python evaluation.py my.model -u
```

Note: Only files with the filetypes `.bin`, `.model` or without any suffix are treated as binary files.

## Download

The optimized German language model, that was trained with this toolkit based on the German Wikipedia (15th May 2015) and German news articles from 2013 (15th May 2015) can be downloaded here:

[german.model](http://cloud.devmount.de/d2bc5672c523b086) [704 MB]

The GermanWordEmbeddings tool and the pretrained language model are completely free to use. If you enjoy it, please consider [donating via Paypal](https://paypal.me/devmount) for further development. :green_heart:
