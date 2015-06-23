import gensim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

model = gensim.models.Word2Vec.load_word2vec_format("model/SG-300-5-NS10-R50.model", binary=True)
# matches = model.most_similar(positive=["Frau"], negative=[], topn=30)
# words = [match[0] for match in matches]
currency = ["China","Yuan","Deutschland","Euro","Daenemark","Krone","England","Pfund","Japan","Yen","Russland","Rubel","USA","Dollar"]
capital  = ["Athen","Griechenland","Bagdad","Irak","Bangkok","Thailand","Berlin","Deutschland","Bern","Schweiz","Hanoi","Vietnam","Helsinki","Finnland","Kairo","Aegypten","Kiew","Ukraine","London","England","Madrid","Spanien","Melbourne","Australien","Moskau","Russland","Oslo","Norwegen","Ottawa","Kanada","Paris","Frankreich","Rom","Italien","Stockholm","Schweden","Teheran","Iran","Tokio","Japan","Washington","USA"]
language = ["China","Chinesisch","Deutschland","Deutsch","England","Englisch","Frankreich","Franzoesisch","Griechenland","Griechisch","Italien","Italienisch","Japan","Japanisch","Korea","Koreanisch","Norwegen","Norwegisch","Polen","Polnisch","Russland","Russisch","Schweden","Schwedisch","Spanien","Spanisch","Ukraine","Ukrainisch"]
words = capital
vectors = [model[word] for word in words]

# pca = PCA(n_components=2)
# vectors2d = pca.fit(vectors).transform(vectors)

tsne = TSNE(n_components=2)
vectors2d = tsne.fit_transform(vectors)

plt.figure()

first = True # color alternation to divide given groups
for point, word in zip(vectors2d , words):
    plt.scatter(
        point[0],
        point[1],
        s=400*len(word),
        c=u'g' if first else u'b',
        marker=r"$ {} $".format(word.replace('_', '\_')),
        edgecolors='none'
    )
    first = not first

plt.title('PCA Darstellung von Wort-Vektoren')

plt.show()
