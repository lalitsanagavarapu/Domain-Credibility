import gensim
from gensim import models
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from gensim.similarities import WmdSimilarity
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedLineDocument
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.downloader as api



w2v_model_300 = api.load("glove-wiki-gigaword-300")
model300=w2v_model_300
model300.save_word2vec_format('model300.bin', binary=True)
print("model 300 tained")
w2v_model_100 = api.load("glove-wiki-gigaword-100")
model100=w2v_model_100
model100.save_word2vec_format('model100.bin', binary=True)
print("model 100 trained")
#filename = 'w2v_model_300'
#pickle.dump(w2v_model_300, open(filename, 'wb'))