import argparse
import data_prepare
import time
from utils.data_processing_utils import *
from models.cnn_lstm_attention import build_cnn_lstm_attention
from evaluator import Evaluator

logger = get_logger("Train sentence sequences Recurrent Convolutional model (LSTM stack over CNN)")

def main():
	parser = argparse.ArgumentParser(description="sentence Hi_CNN model")
	parser.add_argument('--train_flag', action='store_true', help='Train or eval')
	parser.add_argument('--fine_tune', action='store_true', help='Fine tune word embeddings')
	parser.add_argument('--embedding', type=str, default='glove', help='Word embedding type, word2vec, senna or glove')
	parser.add_argument('--embedding_dict', type=str, default='embeddings/glove.6B.50d.txt', help='Pretrained embedding path')
	parser.add_argument('--embedding_dim', type=int, default=64, help='Only useful when embedding is randomly initialised')

	parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
	parser.add_argument('--batch_size', type=int, default=10, help='Number of texts in each batch')
	parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")

	parser.add_argument('--nbfilters', type=int, default=100, help='Num of filters in conv layer')
	parser.add_argument('--filter1_len', type=int, default=5, help='filter length in 1st conv layer')
	parser.add_argument('--filter2_len', type=int, default=3, help='filter length in 2nd conv layer or char conv layer')
	parser.add_argument('--rnn_type', type=str, default='LSTM', help='Recurrent type')
	parser.add_argument('--lstm_units', type=int, default=100, help='Num of hidden units in recurrent layer')

	parser.add_argument('--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop'], help='updating algorithm', default='rmsprop')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
	parser.add_argument('--oov', choices=['random', 'embedding'], default='embedding', help="Embedding for oov word")
	parser.add_argument('--l2_value', type=float, help='l2 regularizer value')
	parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='checkpoint directory')

	parser.add_argument('--train', type=str, help='train file', default='data/reformed/train.tsv')
	parser.add_argument('--dev', type=str, help='dev file', default='data/reformed/valid.tsv')
	parser.add_argument('--test', type=str, help='test file', default='data/reformed/test.tsv')
	parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of dev essay set')
	parser.add_argument('--init_bias', action='store_true', help='init the last layer bias with average score of training data')
	parser.add_argument('--mode', type=str, choices=['mot', 'att', 'merged'], default='att', \
                        help='Mean-over-Time pooling or attention-pooling, or two pooling merged')
	
	args = parser.parse_args()

	train_flag = args.train_flag
	fine_tune = args.fine_tune

	batch_size = args.batch_size
	checkpoint_dir = args.checkpoint_path
	num_epochs = args.num_epochs

	datapaths = [args.train, args.dev, args.test]
	embedding_path = args.embedding_dict
	oov = args.oov
	embedding = args.embedding
	embedd_dim = args.embedding_dim
	prompt_id = args.prompt_id

	(X_train, Y_train, mask_train), (X_dev, Y_dev, mask_dev), (X_test, Y_test, mask_test), \
	vocab, vocab_size, embed_table, overal_maxlen, overal_maxnum, init_mean_value = data_prepare.prepare_data(\
		datapaths, embedding_path, embedding, embedd_dim, prompt_id, args.vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=3)

	if embed_table is not None:
		embedd_dim = embed_table.shape[1]
		embed_table = [embed_table]
        
	max_sentnum = overal_maxnum
	max_sentlen = overal_maxlen

	X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
	X_dev = X_dev.reshape((X_dev.shape[0], X_dev.shape[1]*X_dev.shape[2]))
	X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
	logger.info("X_train shape: %s" % str(X_train.shape))

	model = build_cnn_lstm_attention(args, vocab_size, max_sentnum, max_sentlen, embedd_dim, embed_table, True, init_mean_value)

	# TODO
	modelname="test"

	evl = Evaluator(args.prompt_id, checkpoint_dir, modelname, X_train, X_dev, X_test, Y_train, Y_dev, Y_test)

	logger.info("Initial evaluation: ")
	evl.evaluate(model, -1, print_info=True)
	logger.info("Train model")
	for ii in range(args.num_epochs):
		logger.info('Epoch %s/%s' % (str(ii+1), args.num_epochs))
		start_time = time.time()
		model.fit(X_train, Y_train, batch_size=args.batch_size, nb_epoch=1, verbose=0, shuffle=True)
		tt_time = time.time() - start_time
		logger.info("Training one epoch in %.3f s" % tt_time)
		evl.evaluate(model, ii+1)
		evl.print_info()

	evl.print_final_info()

if __name__ == '__main__':
	main()