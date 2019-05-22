import keras
from keras.models import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, GRU, Dense, merge, concatenate
from keras.layers import TimeDistributed, Conv1D

from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.regularizers import l2

from custom_layers.layers import Attention, ZeroMaskedEntries
from utils.data_processing_utils import get_logger
import time

logger = get_logger("Build model")


def build_cnn_lstm_attention(opts, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (N, L, embedd_dim,
        opts.nbfilters, opts.filter1_len, opts.dropout))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)
    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)
    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    # pooling mode
    if opts.mode == 'mot':
        logger.info("Use mean-over-time pooling on sentence")
        avg_zcnn = TimeDistributed(GlobalAveragePooling1D(), name='avg_zcnn')(zcnn)
    elif opts.mode == 'att':
        logger.info('Use attention-pooling on sentence')
        avg_zcnn = TimeDistributed(Attention(), name='avg_zcnn')(zcnn)
    elif opts.mode == 'merged':
        logger.info('Use mean-over-time and attention-pooling together on sentence')
        avg_zcnn1 = TimeDistributed(GlobalAveragePooling1D(), name='avg_zcnn1')(zcnn)
        avg_zcnn2 = TimeDistributed(Attention(), name='avg_zcnn2')(zcnn)
        avg_zcnn = concatenate([avg_zcnn1, avg_zcnn2], name='avg_zcnn')
    else:
        raise NotImplementedError
    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    if opts.mode == 'mot':
        logger.info('Use mean-over-time pooling on text')
        avg_hz_lstm = GlobalAveragePooling1D(name='avg_hz_lstm')(hz_lstm)
    elif opts.mode == 'att':
        logger.info('Use attention-pooling on text')
        avg_hz_lstm = Attention(name='avg_hz_lstm')(hz_lstm)
    elif opts.mode == 'merged':
        logger.info('Use mean-over-time and attention-pooling together on text')
        avg_hz_lstm1 = GlobalAveragePooling1D(name='avg_hz_lstm1')(hz_lstm)
        avg_hz_lstm2 = Attention(name='avg_hz_lstm2')(hz_lstm)
        avg_hz_lstm = concatenate([avg_hz_lstm1, avg_hz_lstm2], name='avg_hz_lstm')
    else:
        raise NotImplementedError
    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_lstm)

    
    model = Model(inputs=word_input, outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    start_time = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)

    return model