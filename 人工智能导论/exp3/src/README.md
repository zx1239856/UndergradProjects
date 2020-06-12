# Sentiment Analysis

Since the word vectors are giant stuffs unfriendly to limited disk space and bandwidth, they are not included in this project itself.

## How to run the code

+ Download Word Vector from <https://github.com/Embedding/Chinese-Word-Vectors>, more specifically, choose **Sogou News**, the direct link of which is [Sogou](https://pan.baidu.com/s/1tUghuTno5yOvOx4LXA9-wg)

+ Extract Word Vector to **data** dir in **src**, and the filename should be ***sgns.sogou.word***

+ Run the python file via the following command

  ```bash
  python3 train.py -m [MODEL_NAME] -e [NUM_OF_EPOCHS] -b [BATCH_SIZE]
  ### Please refer to the following part for a valid MODEL_NAME
  ```

## Available models

+ REGULAR_CNN	regular CNN implementation
+ MULTI_CHANNEL_CNN_RAN    multi-channel CNN, initialized randomly
+ MULTI_CHANNEL_CNN_PRE    multi-channel CNN, initialized by pre-trained word vector
+ LSTM     basic LSTM implementation
+ BI_LSTM     bi-directional LSTM
+ BASELINE      baseline MLP implementation
+ BASELINE_SIMPLE      baseline MLP, without the need of word vector

## Dependency

+ tensorflow == 1.13.0
+ gensim
+ tqdm
+ other items that fail to import

