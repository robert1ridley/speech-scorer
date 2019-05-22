import gzip
import sys
from gensim.models.word2vec import Word2Vec
import theano
import numpy as np
import logging


def get_logger(name, level=logging.INFO, handler=sys.stdout,
        formatter='%(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def padding_sentence_sequences(index_sequences, scores, max_sentnum, max_sentlen, post_padding=True):

    X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)
    Y = np.empty([len(index_sequences), 1], dtype=np.float32)
    mask = np.zeros([len(index_sequences), max_sentnum, max_sentlen], dtype=theano.config.floatX)

    for i in range(len(index_sequences)):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)

        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            for k in range(length):
                wid = word_ids[k]
                X[i, j, k] = wid
            X[i, j, length:] = 0
            mask[i, j, :length] = 1

        X[i, num:, :] = 0
        Y[i] = scores[i]
    return X, Y, mask


def load_word_embedding_dict(embedding, embedding_path, word_alphabet, logger, embedd_dim=100):
    if embedding == 'word2vec':
        # loading word2vec
        logger.info("Loading word2vec ...")
        word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=False, unicode_errors='ignore')
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim, False
    elif embedding == 'glove':
        # loading GloVe
        logger.info("Loading GloVe ...")
        embedd_dim = -1
        embedd_dict = dict()
        # with gzip.open(embedding_path, 'r') as file:
        with open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    elif embedding == 'senna':
        # loading Senna
        logger.info("Loading Senna ...")
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    else:
        raise ValueError("embedding should choose from [word2vec, senna]")


def build_embedd_table(word_alphabet, embedd_dict, embedd_dim, logger, caseless):
    scale = np.sqrt(3.0 / embedd_dim)
    embedd_table = np.empty([len(word_alphabet), embedd_dim], dtype=theano.config.floatX)
    embedd_table[0, :] = np.zeros([1, embedd_dim])
    oov_num = 0
    # for word, index in word_alphabet.iteritems():
    for word in word_alphabet:
        ww = word.lower() if caseless else word
        # show oov ratio
        if ww in embedd_dict:
            embedd = embedd_dict[ww]
        else:
            embedd = np.random.uniform(-scale, scale, [1, embedd_dim])
            oov_num += 1
        embedd_table[word_alphabet[word], :] = embedd
    oov_ratio = float(oov_num)/(len(word_alphabet)-1)
    logger.info("OOV number =%s, OOV ratio = %f" % (oov_num, oov_ratio))
    return embedd_table


def rescale_tointscore(scaled_scores, set_ids):
    if isinstance(set_ids, int):
        prompt_id = set_ids
        set_ids = np.ones(scaled_scores.shape[0],) * prompt_id
    assert scaled_scores.shape[0] == len(set_ids)
    int_scores = np.zeros((scaled_scores.shape[0], 1))
    for k, i in enumerate(set_ids):
        assert i in range(1, 3)
        if i == 1:
            minscore = 1
            maxscore = 4
        elif i == 2:
            minscore = 1
            maxscore = 4
        else:
            print ("Set ID error")
        int_scores[k] = scaled_scores[k]*(maxscore-minscore) + minscore
    return np.around(int_scores).astype(int)
