import gensim
import os


def load_wv(root):
    pretrained_embeddings_path = os.path.abspath(root)
    path = os.path.join(pretrained_embeddings_path, 'GoogleNews-vectors-negative300.bin')
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    return word2vec

