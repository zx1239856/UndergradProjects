from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from fileio import *
import tensorflow as tf
from tensorflow import keras
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import datetime
import time
import argparse

SENTIMENTS = ['moving', 'sympathetic', 'boring',
              'irritating', 'amusing', 'saddening', 'novel', 'warm']

filters_size = [2, 3, 4, 5, 6, 7, 8, 9, 10]
num_filters = 100


def create_regular_cnn(embedding_mat, img_path='cnn_arch.png'):
    VOCAB_SIZE = embedding_mat.shape[0]
    input_shape = keras.layers.Input(
        shape=(DEFAULT_SENTENCE_LEN,), name="input", )
    static_embedding_ = keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=DEFAULT_SENTENCE_LEN,
                                               weights=[embedding_mat], trainable=False, name='embedding')(input_shape)
    embedding = keras.layers.Reshape(
        (DEFAULT_SENTENCE_LEN, EMBEDDING_SIZE, 1), name="embedding_reshaped")(static_embedding_)
    max_pools = []
    for idx, size in enumerate(filters_size):
        conv = keras.layers.Conv2D(filters=num_filters, kernel_size=(size, EMBEDDING_SIZE), strides=(1, 1), padding="valid",
                                   use_bias=True, kernel_initializer=keras.initializers.TruncatedNormal(0.0, 0.1), bias_initializer=keras.initializers.zeros(), name="conv_%d" % idx)(embedding)
        pool = keras.layers.MaxPool2D(pool_size=(
            DEFAULT_SENTENCE_LEN - size + 1, 1), strides=(1, 1), padding="valid", name="pool_%d" % idx)(conv)
        max_pools.append(pool)
    merged = keras.layers.concatenate(max_pools, axis=1)
    merged = keras.layers.Flatten()(merged)
    dropout = keras.layers.Dropout(rate=0.2)(merged)
    out = keras.layers.Dense(8, activation=keras.activations.softmax, use_bias=True, kernel_initializer=keras.initializers.glorot_normal(
    ), bias_initializer=keras.initializers.zeros(), name="dense_out")(dropout)
    model = keras.Model(input_shape, out)
    keras.utils.plot_model(model, img_path, show_shapes=False)
    return model


def create_multi_channel_cnn(embedding_mat, img_path='cnn_arch.png', custom_embedding=None):
    VOCAB_SIZE = embedding_mat.shape[0]
    input_shape = keras.layers.Input(
        shape=(DEFAULT_SENTENCE_LEN,), name="input", )
    static_embedding_ = keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=DEFAULT_SENTENCE_LEN,
                                               weights=[embedding_mat], trainable=False, name='embedding')(input_shape)
    static_embedding = keras.layers.Reshape(
        (DEFAULT_SENTENCE_LEN, EMBEDDING_SIZE, 1), name="embedding_reshaped")(static_embedding_)
    dynamic_embedding_ = None
    if(custom_embedding is not None):
        dynamic_embedding_ = keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, weights=[
                                                    custom_embedding], input_length=DEFAULT_SENTENCE_LEN, trainable=True, name='embedding_d')(input_shape)
    else:
        dynamic_embedding_ = keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, embeddings_initializer=keras.initializers.random_normal(
            0.0, 0.1), input_length=DEFAULT_SENTENCE_LEN, trainable=True, name='embedding_d')(input_shape)
    dynamic_embedding = keras.layers.Reshape(
        (DEFAULT_SENTENCE_LEN, EMBEDDING_SIZE, 1), name="embedding_d_reshaped")(dynamic_embedding_)
    embedding = keras.layers.concatenate(
        [static_embedding, dynamic_embedding], axis=-1)
    max_pools = []
    for idx, size in enumerate(filters_size):
        conv = keras.layers.Conv2D(filters=num_filters, kernel_size=(size, EMBEDDING_SIZE), strides=(1, 1), padding="valid",
                                   use_bias=True, kernel_initializer=keras.initializers.TruncatedNormal(0.0, 0.1), bias_initializer=keras.initializers.zeros(), name="conv_%d" % idx)(embedding)
        pool = keras.layers.MaxPool2D(pool_size=(
            DEFAULT_SENTENCE_LEN - size + 1, 1), strides=(1, 1), padding="valid", name="pool_%d" % idx)(conv)
        max_pools.append(pool)
    merged = keras.layers.concatenate(max_pools, axis=1)
    merged = keras.layers.Flatten()(merged)
    dropout = keras.layers.Dropout(rate=0.2)(merged)
    out = keras.layers.Dense(8, activation=keras.activations.softmax, use_bias=True, kernel_initializer=keras.initializers.glorot_normal(
    ), bias_initializer=keras.initializers.zeros(), name="dense_out")(dropout)
    model = keras.Model(input_shape, out)
    keras.utils.plot_model(model, img_path, show_shapes=False)
    return model


def create_simple_lstm(embedding_mat, img_path='lstm_arch.png'):
    VOCAB_SIZE = embedding_mat.shape[0]
    input_shape = keras.layers.Input(
        shape=(DEFAULT_SENTENCE_LEN,), name="input", )
    static_embedding_ = keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=DEFAULT_SENTENCE_LEN,
                                               weights=[embedding_mat], trainable=False, name='embedding', mask_zero=True)(input_shape)
    lstm = keras.layers.LSTM(128, activation='tanh',
                             name='lstm')(static_embedding_)
    #lstm = keras.layers.CuDNNLSTM(128)(static_embedding_)
    dropout = keras.layers.Dropout(rate=0.2)(lstm)
    out = keras.layers.Dense(8, activation=keras.activations.softmax, use_bias=True, kernel_initializer=keras.initializers.glorot_normal(
    ), bias_initializer=keras.initializers.zeros(), name="dense_out")(dropout)
    model = keras.Model(input_shape, out)
    keras.utils.plot_model(model, img_path, show_shapes=True)
    return model

def create_bidirectional_lstm(embedding_mat, img_path='lstm_arch.png'):
    VOCAB_SIZE = embedding_mat.shape[0]
    input_shape = keras.layers.Input(
        shape=(DEFAULT_SENTENCE_LEN,), name="input", )
    static_embedding_ = keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=DEFAULT_SENTENCE_LEN,
                                               weights=[embedding_mat], trainable=False, name='embedding', mask_zero=True)(input_shape)
    lstm = keras.layers.LSTM(128, activation='tanh',
                             name='lstm')(static_embedding_)
    lstm_r = keras.layers.LSTM(
        128, activation='tanh', name='lstm_r', go_backwards=True)(static_embedding_)
    merged = keras.layers.Add()([lstm, lstm_r])
    #lstm = keras.layers.CuDNNLSTM(128)(static_embedding_)
    dropout = keras.layers.Dropout(rate=0.2)(merged)
    out = keras.layers.Dense(8, activation=keras.activations.softmax, use_bias=True, kernel_initializer=keras.initializers.glorot_normal(
    ), bias_initializer=keras.initializers.zeros(), name="dense_out")(dropout)
    model = keras.Model(input_shape, out)
    keras.utils.plot_model(model, img_path, show_shapes=True)
    return model


def create_baseline_mlp_simple(img_path='mlp.png'):
    input_shape = keras.layers.Input(
        shape=(DEFAULT_SENTENCE_LEN,), name="input", )
    # static_embedding_ = keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=DEFAULT_SENTENCE_LEN,
    #                                           weights=[embedding_mat], trainable=False, name='embedding', mask_zero=False)(input_shape)
    dense1 = keras.layers.Dense(
        512, activation='tanh', kernel_initializer=keras.initializers.glorot_normal())(input_shape)
    dense2 = keras.layers.Dense(
        256, activation='tanh', kernel_initializer=keras.initializers.glorot_normal())(dense1)
    dense3 = keras.layers.Dense(
        64, activation='tanh', kernel_initializer=keras.initializers.glorot_normal())(dense2)
    out = keras.layers.Dense(8, activation=keras.activations.softmax, use_bias=True, kernel_initializer=keras.initializers.glorot_normal(
    ), bias_initializer=keras.initializers.zeros(), name="dense_out")(dense3)
    model = keras.Model(input_shape, out)
    keras.utils.plot_model(model, img_path, show_shapes=True)
    return model


def create_baseline_mlp(embedding_mat, img_path='mlp.png'):
    VOCAB_SIZE = embedding_mat.shape[0]
    input_shape = keras.layers.Input(
        shape=(DEFAULT_SENTENCE_LEN,), name="input", )
    static_embedding_ = keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=DEFAULT_SENTENCE_LEN,
                                               weights=[embedding_mat], trainable=False, name='embedding', mask_zero=False)(input_shape)

    def impl_sum(x):
        return keras.backend.sum(x, axis=1)
    static_embedding_ = keras.layers.Lambda(impl_sum)(static_embedding_)
    dense1 = keras.layers.Dense(
        200, activation='tanh', kernel_initializer=keras.initializers.glorot_normal())(static_embedding_)
    dense2 = keras.layers.Dense(
        100, activation='tanh', kernel_initializer=keras.initializers.glorot_normal())(dense1)
    dense3 = keras.layers.Dense(
        50, activation='tanh', kernel_initializer=keras.initializers.glorot_normal())(dense2)
    out = keras.layers.Dense(8, activation=keras.activations.softmax, use_bias=True, kernel_initializer=keras.initializers.glorot_normal(
    ), bias_initializer=keras.initializers.zeros(), name="dense_out")(dense3)
    model = keras.Model(input_shape, out)
    keras.utils.plot_model(model, img_path, show_shapes=True)
    return model


def multi_label_acc(y_true, y_pred):
    mat = keras.backend.zeros_like(y_pred)
    max_val = keras.backend.max(y_pred, axis=-1, keepdims=True)
    comp = keras.backend.equal(mat, y_pred - max_val)
    comp = keras.backend.batch_dot(keras.backend.cast(
        comp, keras.backend.floatx()), y_true, axes=-1)
    #comp = keras.backend.dot(y_true, y_pred)
    return keras.backend.cast(keras.backend.any(comp, axis=-1), keras.backend.floatx())


if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epoch', type=int, help='Number of epochs', default=30)
    parser.add_argument('-b', '--batch', type=int, help='Batch size', default=256)
    parser.add_argument('-m', '--model', help='The type of model to train', default="REGULAR_CNN")
    args = parser.parse_args()
    print("Model %s, batch size: %d, epoch num: %d" % (args.model, args.batch, args.epoch))
    need_word_vec = True
    if(args.model == "BASELINE_SIMPLE"):
        need_word_vec = False
    if(need_word_vec):
        word_vec2 = KeyedVectors.load('embedding.model')
        words, word_vec = loadWordVec("data/sgns.sogou.word")
    word_list, y, _ = loadCorpus("data/sinanews.train")
    idx2word, word2idx = buildVocabulary(word_list)
    if(need_word_vec):
        embedding_mat = buildEmbeddingMatrix(idx2word, word2idx, word_vec)
        embedding_mat2 = buildEmbeddingMatrix(idx2word, word2idx, word_vec2)
    X = []
    # convert word_list to idx_list for training
    for sentence in word_list:
        X.append(sentenceToSeq(sentence, word2idx))
    X = np.array(X)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.15, random_state=3)
    X_test = []
    word_list, y_test, y_norm = loadCorpus("data/sinanews.test")
    for sentence in word_list:
        X_test.append(sentenceToSeq(sentence, word2idx))
    X_test = np.array(X_test)

    if(args.model == "REGULAR_CNN"):
        model = create_regular_cnn(embedding_mat)
    elif(args.model == "MULTI_CHANNEL_CNN_RAN"):
        model = create_multi_channel_cnn(embedding_mat, 'cnn_arch.png')
    elif(args.model == "MULTI_CHANNEL_CNN_PRE"):
        model = create_multi_channel_cnn(embedding_mat, 'cnn_arch.png', embedding_mat2)
    elif(args.model == "LSTM"):
        model = create_simple_lstm(embedding_mat)
    elif(args.model == "BI_LSTM"):
        model = create_bidirectional_lstm(embedding_mat)
    elif(args.model == "BASELINE"):
        model = create_baseline_mlp(embedding_mat)
    elif(args.model == "BASELINE_SIMPLE"):
        model = create_baseline_mlp_simple()
    else:
        exit("Model not supported!")
    model.summary()
    tensorboard = keras.callbacks.TensorBoard(
        log_dir="logs/{}".format(time.time()))
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=[multi_label_acc])
    history = model.fit(X_train, y_train, epochs=args.epoch, batch_size=args.batch, validation_data=(
        X_eval, y_eval), verbose=1, callbacks=[tensorboard])
    results = model.evaluate(X_test, y_test)
    test = model.predict(X_test)
    corr = []
    process_y_truth = []
    process_y_pred = []
    for i in range(0, len(test)):
        pred = np.zeros(len(test[i]))
        pos = np.argmax(test[i])
        pred[pos] = 1
        truth = np.array(y_test[i])
        process_y_pred.append(pos)
        if(truth[pos] >= 1):
            process_y_truth.append(pos)
        else:
            process_y_truth.append(np.argmax(truth))
        corr.append(pearsonr(test[i], y_norm[i])[0])
    print("Correlation: %f, F1-Score: %f" % (np.average(corr), f1_score(process_y_truth, process_y_pred, average='macro')))
    print(results)
