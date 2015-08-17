# start script in new folder
mkdir word2vec
cd word2vec/
mkdir corpus
mkdir model

# get scripts
wget https://raw.githubusercontent.com/devmount/GermanWordEmbeddings/master/preprocessing.py
wget https://raw.githubusercontent.com/devmount/GermanWordEmbeddings/master/training.py
wget https://raw.githubusercontent.com/devmount/GermanWordEmbeddings/master/evaluation.py

# build news corpus
for i in 2007 2008 2009 2010 2011 2012 2013; do
	wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.$i.de.shuffled.gz
	gzip -d news.$i.de.shuffled.gz
	python preprocessing.py news.$i.de.shuffled corpus/news.$i.de.shuffled.corpus -psub
done
rm news*

# build wikipedia corpus
wget http://download.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2
wget http://medialab.di.unipi.it/Project/SemaWiki/Tools/WikiExtractor.py
python WikiExtractor.py -c -b 25M -o extracted dewiki-latest-pages-articles.xml.bz2
find extracted -name '*bz2' \! -exec bzip2 -k -c -d {} \; > dewiki.xml
printf "Number of articles: "
grep -o "<doc" dewiki.xml | wc -w
sed -i 's/<[^>]*>//g' dewiki.xml
rm -rf extracted
python preprocessing.py dewiki.xml corpus/dewiki.corpus -psub
rm dewiki.xml
# only keep .bigram corpus files (preprocessing.py -b creates additional .bigram files to normal .corpus files)
rm corpus/*.corpus

# train model with vector size 300, window size 5, 10 negative samples and word min count of 50
python training.py corpus/ model/my.model -s 300 -w 5 -n 10 -m 50

# evaluation with top 10 results
python evaluation.py model/my.model -u -t 10
