from fileio import loadCorpus
import gensim

class SentenceIter(object):
    def __init__(self, word_list):
        self.word_list = word_list
    def __iter__(self):
        for i in self.word_list:
            yield i.split(' ')

if(__name__ == "__main__"):
    word_list, y = loadCorpus("data/sinanews.train")
    model = gensim.models.Word2Vec(word_list)
    sentences = SentenceIter(word_list)
    model = gensim.models.Word2Vec(sentences, size = 300, min_count=3, workers=12)
    model.save('embedding.model')