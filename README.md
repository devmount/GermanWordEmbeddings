# GermanWordEmbeddings
There has been a lot of research into training of word embeddings on English corpora. This toolkit applies deep learning via [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) on German corpora to train and evaluate German models.

1. [Obtaining corpora](#obtention)
2. [Preprocessing](#preprocessing)
3. [Training models](#training)
3. [Vocabulary](#vocabulary)
3. [Evaluation](#evaluation)

## Obtaining corpora <a name="obtention"></a>
There are multiple possibilities for an obtention of huge German corpora that are public and free to use. For example the German Wikipedia:
```shell
wget http://download.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2
```
Or shuffled German news in 2007 to 2013:
```shell
for i in 2007 2008 2009 2010 2011 2012 2013; do
  wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.$i.de.shuffled.gz
done
```

## Preprocessing <a name="preprocessing"></a>
This Tool preprocesses the raw wikipedia XML corpus with the WikipediaExtractor (a Python Script from Giuseppe Attardi to filter a Wikipedia XML Dump) and some shell instructions to filter all XML tags and quotations:
```shell
wget http://medialab.di.unipi.it/Project/SemaWiki/Tools/WikiExtractor.py
python WikiExtractor.py -c -b 25M -o extracted dewiki-latest-pages-articles.xml.bz2
find extracted -name '*bz2' \! -exec bzip2 -k -c -d {} \; > dewiki.xml
sed -i 's/<[^>]*>//g' dewiki.xml
sed -i 's|["'\''„“‚‘]||g' dewiki.xml
rm -rf extracted
```
The German news are already containing one sentence per line and don't have any XML overhead. Only the quotation has to be removed:
```shell
for i in 2007 2008 2009 2010 2011 2012 2013; do
  gzip -d news.$i.de.shuffled.gz
  sed -i 's|["'\''„“‚‘]||g' news.$i.de.shuffled
done
```

Afterwards, the `preprocessing.py` script can be called on all the corpus files with the following options:

flag               | description
------------------ | -----------------------------------------------------
-h, --help         | show this help message and exit
-p, --punctuation  | filter punctuation tokens
-s, --stopwords    | filter stop word tokens
-u, --umlauts      | replace german umlauts with their respective digraphs
-b, --bigram       | detect and process common bigram phrases


## Training models <a name="training"></a>
## Vocabulary <a name="vocabulary"></a>
## Evaluation <a name="evaluation"></a>
