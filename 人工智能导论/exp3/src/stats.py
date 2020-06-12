from fileio import *
from train import SENTIMENTS
import numpy as np

def stat_all_set(file_name):
    word_list, y, _ = loadCorpus(file_name)
    print("Total: %d" % len(y))
    cnt = np.zeros(8, dtype=int)
    for item in y:
        cnt += np.array(item, dtype=int)
    print("SENTIMENTS STAT: ")
    cnt = np.array(cnt, dtype=float) / len(y);
    for item in zip(cnt, SENTIMENTS):
        print(item)

if(__name__ == "__main__"):
    stat_all_set("data/sinanews.train")
    stat_all_set("data/sinanews.test")