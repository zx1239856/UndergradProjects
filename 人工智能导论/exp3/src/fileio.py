import tqdm
import re
import numpy as np
from itertools import islice
from collections import Counter

re_num = re.compile('[0-9]+')
EMBEDDING_SIZE = 300
DEFAULT_SENTENCE_LEN = 1000

def buildVocabulary(corpus, word_vec_set = None):
    word_list = []
    for item in corpus:
        word_list.extend(item.split(' '))
    freq = Counter(word_list)
    idx2word = ['<PAD>', '<NUL>']
    word2idx = {'<PAD>': 0, '<NUL>': 1}
    count = 2
    num_has_vec = 0
    for w, c in tqdm.tqdm(freq.most_common()):
        if(c > 1):
            idx2word.append(w)
            word2idx[w] = count
            count += 1
            if(word_vec_set is not None and w in word_vec_set):
                num_has_vec += 1
    if(word_vec_set is not None):
        print("There are %d words in total, but only %d are used, and %d have embedding vector" % (len(freq), len(idx2word), num_has_vec))
    else:
        print("There are %d words in total, but only %d are used" % (len(freq), len(idx2word)))
    return idx2word, word2idx

def buildEmbeddingMatrix(idx2word, word2idx, word2vec):
    embedding = np.zeros([len(idx2word), EMBEDDING_SIZE])
    rnd = np.random.random_sample((EMBEDDING_SIZE))
    for w, i in tqdm.tqdm(word2idx.items()):
        if(isinstance(word2vec, dict)):
            embedding[i, :] = word2vec.get(w, rnd)
        else:
            # gensim vector
            try:
                embedding[i, :] = word2vec[w]
            except:
                embedding[i, :] = rnd
    embedding[0, :] = np.zeros(EMBEDDING_SIZE)
    return embedding

def loadCorpus(filename):
    word_list = []
    sentiments = []
    sentiments_normed = []
    cnt = 0
    tot_len = 0
    max_len = 0
    with open(filename, "r", encoding="utf8") as f:
        for line in tqdm.tqdm(f.readlines()):
            line = line.strip('\n').split('\t')
            sentiment = [int(re_num.findall(item)[0]) for item in line[1].split(' ')]
            max_sentiment = max(sentiment[1:])
            label = np.zeros(len(sentiment) - 1)
            label_normed = np.array(sentiment[1:]) / max_sentiment
            for i in range(1, len(sentiment)):
                if(sentiment[i] == max_sentiment):
                    label[i - 1] = 1
            sentiments.append(label)
            sentiments_normed.append(label_normed)
            content = line[2]
            len_line = len(content.split(' '))
            tot_len += len_line 
            if(len_line > max_len):
                max_len = len_line
            cnt += 1
            word_list.append(content)
    print("Corpus info: average length %f, max length: %d"%(tot_len / cnt, max_len))
    return word_list, np.array(sentiments), np.array(sentiments_normed)

def sentenceToSeq(sentence, word2idx, length = DEFAULT_SENTENCE_LEN):
    entry = [word2idx.get(word, 1) for word in sentence.split(' ')]
    if(len(entry) < length):
        entry.extend([0] * (length - len(entry)))
    else:
        entry = entry[:length]
    return np.array(entry)
     

def loadWordVec(filename):
    print("Loading word vector...")
    words = set()
    word2vec = {}
    with open(filename, "r", encoding="utf8") as f:
        for line in tqdm.tqdm(islice(f, 1, None)):
            line = line.strip().split()
            word = line[0]
            words.add(word)
            word2vec[word] = np.array(line[1:], dtype=np.float32)
    return words, word2vec