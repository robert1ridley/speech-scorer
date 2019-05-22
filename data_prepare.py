import reader
from utils.data_processing_utils import get_logger
from utils.data_processing_utils import padding_sentence_sequences, load_word_embedding_dict, build_embedd_table
logger = get_logger("Loading data...")

def prepare_data(datapaths, embedding_path=None, embedding='word2vec', embedd_dim=100, prompt_id=1, vocab_size=0, tokenize_text=True, \
                         to_lower=True, sort_by_len=False, vocab_path=None, score_index=3):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x, train_y), (dev_x, dev_y), (test_x, test_y), vocab, overal_maxlen, overal_maxnum = \
        reader.get_data(datapaths, prompt_id, vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=6)

    X_train, y_train, mask_train = padding_sentence_sequences(train_x, train_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_dev, y_dev, mask_dev = padding_sentence_sequences(dev_x, dev_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_test, y_test, mask_test = padding_sentence_sequences(test_x, test_y, overal_maxnum, overal_maxlen, post_padding=True)

    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)

    Y_train = reader.get_model_friendly_scores(y_train, prompt_id)
    Y_dev = reader.get_model_friendly_scores(y_dev, prompt_id)
    Y_test = reader.get_model_friendly_scores(y_test, prompt_id)
    scaled_train_mean = reader.get_model_friendly_scores(train_mean, prompt_id)

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' % 
                (str(train_mean), str(train_std), str(scaled_train_mean)))

    if embedding_path:
        embedd_dict, embedd_dim, _ = load_word_embedding_dict(embedding, embedding_path, vocab, logger, embedd_dim)
        embedd_matrix = build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)
    else:
        embedd_matrix = None
    
    return (X_train, Y_train, mask_train), (X_dev, Y_dev, mask_dev), (X_test, Y_test, mask_test), vocab, len(vocab), embedd_matrix, overal_maxlen, overal_maxnum, scaled_train_mean