import codecs
import sys
import re
import nltk
import numpy as np
from utils.data_processing_utils import get_logger
logger = get_logger("Loading data...")
url_replacer = '<url>'
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

MAX_SENTLEN = 50
MAX_SENTNUM = 100


asap_ranges = {
    0: (1, 4),
    1: (1, 4),
    2: (1, 4)
}


def get_model_friendly_scores(scores_array, prompt_id_array):
    arg_type = type(prompt_id_array)
    assert arg_type in {int, list, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scaled_scores_array = (scores_array - low) / (high - low)
    else:
        num_scores = scores_array.shape[0]
        scaled_scores_array = np.zeros((num_scores, 1))
        for i in range(len(prompt_id_array)):
            low, high = asap_ranges[prompt_id_array[i]]
            current_score = scores_array[i, 0]
            scaled_scores_array[i, 0] = (current_score - low) / (high - low)
    assert np.all(scaled_scores_array >= 0) and np.all(scaled_scores_array <= 1)
    return scaled_scores_array


def create_vocab(file_path, prompt_id, vocab_size, tokenize_text, to_lower):
    logger.info('Creating vocabulary from: ' + file_path)
    total_words, unique_words = 0, 0
    word_freqs = {}
    with codecs.open(file_path, mode='r', encoding='UTF8', errors='replace') as input_file:
        next(input_file)
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = tokens[0]
            essay_set = tokens[1]
            content = tokens[2].strip()
            score = tokens[3]
            if int(essay_set) == prompt_id or prompt_id <= 0:
                if tokenize_text:
                    content = text_tokenizer(content, True, True, True)
                if to_lower:
                    content = [w.lower() for w in content]
                for word in content:
                    try:
                        word_freqs[word] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[word] = 1
                    total_words += 1
    logger.info('  %i total words, %i unique words' % (total_words, unique_words))
    import operator
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    if vocab_size <= 0:
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    vcb_len = len(vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1
    return vocab


def shorten_sentence(sent, max_sentlen):
    new_tokens = []
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    if len(tokens) > max_sentlen:
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = len(tokens) / max_sentlen
            num = int(round(num))
            k_indexes = [(i+1)*max_sentlen for i in range(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])
        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = len(token) / max_sentlen
                num = int(np.ceil(num))
                s_indexes = [(i+1)*max_sentlen for i in range(num)]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in range(len_s-1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
            return [tokens]
    return new_tokens


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def tokenize_to_sentences(text, max_sentlength, create_vocab_flag=False):
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
    processed_sents = []
    for sent in sents:
        if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
            s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)
            ss = " ".join(s)
            ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)
            processed_sents.extend(ssL)
        else:
            processed_sents.append(sent)
    if create_vocab_flag:
        sent_tokens = [tokenize(sent) for sent in processed_sents]
        tokens = [w for sent in sent_tokens for w in sent]
        return tokens
    sent_tokens = []
    for sent in processed_sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sentlength)
        sent_tokens.extend(shorten_sents_tokens)
    return sent_tokens


def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text


def is_number(token):
    return bool(num_regex.match(token))


def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    text = replace_url(text)
    text = text.replace(u'"', u'')
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)

    tokens = tokenize(text)
    if tokenize_sent_flag:
        text = " ".join(tokens)
        sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
        return sent_tokens
    else:
        raise NotImplementedError


def read_dataset(file_path, prompt_id, vocab, to_lower, score_index=3):
    logger.info('Reading dataset from: ' + file_path)
    data_x, data_y, prompt_ids = [], [], []
    num_hit, unk_hit, total = 0., 0., 0.
    max_sentnum = -1
    max_sentlen = -1
    with codecs.open(file_path, mode='r', encoding='UTF8', errors='replace') as input_file:
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = tokens[0]
            essay_set = tokens[1]
            content = tokens[2].strip()
            score = tokens[score_index]
            if int(essay_set) == prompt_id or prompt_id <= 0:
                sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
                if to_lower:
                    sent_tokens = [[w.lower() for w in s] for s in sent_tokens]
                sent_indices = []
                indices = []
                for sent in sent_tokens:
                    length = len(sent)
                    if max_sentlen < length:
                        max_sentlen = length

                    for word in sent:
                        if is_number(word):
                            indices.append(vocab['<num>'])
                            num_hit += 1
                        elif word in vocab:
                            indices.append(vocab[word])
                        else:
                            indices.append(vocab['<unk>'])
                            unk_hit += 1
                        total += 1
                    sent_indices.append(indices)
                    indices = []
                data_x.append(sent_indices)
                try:
                    data_y.append(int(score))
                except ValueError:
                    print("error with essay id: {}".format(essay_id))
                    raise ValueError
                prompt_ids.append(essay_set)
                if max_sentnum < len(sent_indices):
                    max_sentnum = len(sent_indices)
    logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
    return data_x, data_y, prompt_ids, max_sentlen, max_sentnum


def get_data(paths, prompt_id, vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=3):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    logger.info("Prompt id is %s" % prompt_id)
    if not vocab_path:
        vocab = create_vocab(train_path, prompt_id, vocab_size, tokenize_text, to_lower)
        if len(vocab) < vocab_size:
            logger.warning('The vocabulary includes only %i words (less than %i)' % (len(vocab), vocab_size))
        else:
            assert vocab_size == 0 or len(vocab) == vocab_size
    else:
        vocab = load_vocab(vocab_path)
        if len(vocab) != vocab_size:
            logger.warning('The vocabulary includes %i words which is different from given: %i' % (len(vocab), vocab_size))
    logger.info('  Vocab size: %i' % (len(vocab)))

    train_x, train_y, train_prompts, train_maxsentlen, train_maxsentnum = read_dataset(train_path, prompt_id, vocab, to_lower)
    dev_x, dev_y, dev_prompts, dev_maxsentlen, dev_maxsentnum = read_dataset(dev_path, prompt_id, vocab, to_lower)
    test_x, test_y, test_prompts, test_maxsentlen, test_maxsentnum = read_dataset(test_path, prompt_id, vocab,  to_lower)

    overal_maxlen = max(train_maxsentlen, dev_maxsentlen, test_maxsentlen)
    overal_maxnum = max(train_maxsentnum, dev_maxsentnum, test_maxsentnum)

    logger.info("Training data max sentence num = %s, max sentence length = %s" % (train_maxsentnum, train_maxsentlen))
    logger.info("Dev data max sentence num = %s, max sentence length = %s" % (dev_maxsentnum, dev_maxsentlen))
    logger.info("Test data max sentence num = %s, max sentence length = %s" % (test_maxsentnum, test_maxsentlen))
    logger.info("Overall max sentence num = %s, max sentence length = %s" % (overal_maxnum, overal_maxlen))
    return (train_x, train_y), (dev_x, dev_y), (test_x, test_y), vocab, overal_maxlen, overal_maxnum
