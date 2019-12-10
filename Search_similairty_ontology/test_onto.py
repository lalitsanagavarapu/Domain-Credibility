
from nltk.corpus import wordnet
import pronto
from scipy import spatial

ms = pronto.Ontology("securityontology.owl")
# model.wv.similarity("a","b")
# index2word_set = set(mod)
word = "weapon"

syn = wordnet.synsets(word)

if syn == []:
    keyword = word
else:
    w = syn[0]
    keyword = w.hypernyms()[0].lemmas()[0].name()
    print(keyword)

for term in ms:
    print (term.name)


