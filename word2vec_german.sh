# # start script in new folder
# mkdir word2vec
# cd word2vec
# mkdir corpus
# mkdir model
#
# # build news corpus
# for i in 2007 2008 2009 2010 2011 2012 2013; do
# 	wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.$i.de.shuffled.gz
# 	gzip -d news.$i.de.shuffled.gz
# 	sed -i 's|["'\''„“‚‘]||g' news.$i.de.shuffled
# 	python preprocessing.py news.$i.de.shuffled corpus_ps/news.$i.de.shuffled.corpus -ps
# done
# rm news*
#
# # build wikipedia corpus
# wget http://download.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2
# wget http://medialab.di.unipi.it/Project/SemaWiki/Tools/WikiExtractor.py
# python WikiExtractor.py -c -b 25M -o extracted dewiki-latest-pages-articles.xml.bz2
# find extracted -name '*bz2' \! -exec bzip2 -k -c -d {} \; > dewiki.xml
# printf "Number of articles: "
# grep -o "<doc" dewiki.xml | wc -w
# sed -i 's/<[^>]*>//g' dewiki.xml
# sed -i 's|["'\''„“‚‘]||g' dewiki.xml
# rm -rf extracted
# python preprocessing.py dewiki.xml corpus_ps/dewiki.corpus -ps
# rm dewiki.xml
#
# # training
# { time python training.py corpus/ model/corpus_SG-52-5.model -s 52 -w 5; } 2> model/corpus_SG-52-5.model.result
# { time python training.py corpus_ps/ model/corpus-ps_SG-52-5.model -s 52 -w 5; } 2> model/corpus-ps_SG-52-5.model.result
# { time python training.py corpus_psu/ model/corpus-psu_SG-52-5.model -s 52 -w 5; } 2> model/corpus-psu_SG-52-5.model.result
# { time python training.py corpus_psub/ model/corpus-psub_SG-52-5.model -s 52 -w 5; } 2> model/corpus-psub_SG-52-5.model.result
#
# evaluation
# for m in SG-300-10; do
{ time python evaluation.py model/corpus_SG-52-5.model; } 2>> model/corpus_SG-52-5.model.result
# 	{ time python evaluation.py model/corpus-ps_SG-52-5.model; } 2>> model/corpus-ps_SG-52-5.model.result
# 	{ time python evaluation.py model/corpus-psu_SG-52-5.model; } 2>> model/corpus-psu_SG-52-5.model.result
# 	{ time python evaluation.py model/corpus-psub_SG-52-5.model; } 2>> model/corpus-psub_SG-52-5.model.result
# done
