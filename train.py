import os
import sys
import time
import platform
import cPickle as pkl
import random
import copy
import argparse
import logging

import subprocess

import numpy

from theano import config

sys.path.append(".")
from lib import *
from build import *

sys.setrecursionlimit(1500)
#config.optimizer='fast_compile'
#config.exception_verbosity='high'

def display(msg, logger):
	print msg
	logger.info(msg)
	
def train(options):
	if not options.folder:
		options.folder = "workshop"
	if not os.path.exists(options.folder):
		os.mkdir(options.folder)
	
	logging.basicConfig(level = logging.DEBUG,
						format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s",
						filename = os.path.join(options.folder, options.file_log),
						filemode = "w")
	logger = logging.getLogger()

	logger.info("python %s" %(" ".join(sys.argv)))
	
	#################################################################################
	start_time = time.time()
	
	msg = "Loading dicts from %s..." %(options.file_dic)
	display(msg, logger)
	word2idx, label2idx = dicfold(options.file_dic)
	
	msg = "Loading data from %s..." %(options.file_train)
	display(msg, logger)
	train_x, train_y, train_z = datafold(options.file_train)
	
	msg = "Loading data from %s..." %(options.file_valid)
	display(msg, logger)
	valid_x, valid_y, valid_z = datafold(options.file_valid)
	
	msg = "Loading data from %s..." %(options.file_test)
	display(msg, logger)
	test_x, test_y, test_z = datafold(options.file_test)
	
	end_time = time.time()
	msg = "Loading data time: %f seconds" %(end_time - start_time)
	display(msg, logger)
	
	options.dic= word2idx
	options.size_vocab = len(word2idx)
	options.num_class = len(label2idx)
	
	if options.validFreq == -1:
		options.validFreq = (len(train_x) + options.batch_size - 1) / options.batch_size
	
	if options.saveFreq == -1:
		options.saveFreq = (len(train_x) + options.batch_size - 1) / options.batch_size
	
	msg = "\n#inst in train: %d\n"	\
		  "#inst in dev %d\n"		\
		  "#inst in test %d\n"		\
		  "#vocab: %d\n"			\
		  "#label: %d" %(len(train_x), len(valid_x), len(test_x), options.size_vocab, options.num_class)
	display(msg, logger)
	
	#################################################################################
	start_time = time.time()
	msg = 'Building model...'
	display(msg, logger)
	
	numpy.random.seed(options.seed)
	random.seed(options.seed)
	
	params, tparams, use_noise, x, mask_x, emb_x, y, mask_y, emb_y, z, pred_prob, pred, log_cost, cost, grad_norm, grads = build_model(options)

	lr = tensor.scalar(dtype=config.floatX)
	
	f_grad_shared, f_update, f_log_cost, f_grad_norm, f_pred_prob, f_pred = build_optimizer(lr, tparams, x, y, mask_x, mask_y, z, pred_prob, pred, log_cost, cost, grad_norm, grads, options)
	
	end_time = time.time()
	msg = "#Params: %d, Building time: %f seconds" %(get_model_size(params), end_time - start_time)
	display(msg, logger)
	
	#################################################################################
	msg = 'Optimization...'
	display(msg, logger)
	
	kf_test = get_minibatches_idx(len(test_x), options.batch_size)
	kf_valid = get_minibatches_idx(len(valid_x), options.batch_size)
	beta_train = get_minibatches_idx(len(train_x), options.batch_size)
	
	estop = False
	history_errs = []
	valid_acc_record = []
	test_acc_record = []
	best_p = None
	
	bad_counter = 0
	wait_counter = 0
	wait_N = 1
	lr_change_list = []
	
	n_updates = 0
	best_epoch_num = 0
	
	start_time = time.time()
	
	for e in xrange(options.nepochs):
		kf_train = get_minibatches_idx(len(train_x), options.batch_size, shuffle = True)
		
		n_samples = 0
		
		for _, train_index in kf_train:
			n_updates += 1
			n_samples += len(train_index)
			
			use_noise.set_value(1.)
			
			x = [train_x[t] for t in train_index]
			y = [train_y[t] for t in train_index]
			z = [train_z[t] for t in train_index]
			
			x_, mask_x_ = prepare_data(x)
			y_, mask_y_ = prepare_data(y)
			z_ = numpy.array(z)
			
			if x_ is None:
				msg = "Minibatch with zero sample under length %d" %(options.maxlen)
				display(msg, logger)
				n_updates -= 1
				continue
			
			disp_start = time.time()

			cost = f_grad_shared(x_, y_, mask_x_, mask_y_, z_)
			f_update(options.lr)
			
			disp_end = time.time()
			
			if numpy.isnan(cost) or numpy.isinf(cost):
				msg = "NaN detected"
				display(msg, logger)
			
			if numpy.mod(n_updates, options.dispFreq) == 0:
				msg = "Epoch: %d, Update: %d, Cost: %f, Grad: %f, Time: %.2f sec" %(e, n_updates, cost, f_grad_norm(x_, y_, mask_x_, mask_y_, z_), (disp_end-disp_start))
				display(msg, logger)
				
			if numpy.mod(n_updates, options.saveFreq) == 0:
				msg = "Saving..."
				display(msg, logger)
				if best_p is not None:
					params = best_p
				else:
					params = unzip(tparams)
				
				numpy.savez(os.path.join(options.folder, options.saveto), **params)
				pkl.dump(options, open('%s.pkl' %os.path.join(options.folder, options.saveto), 'wb'))
				msg = "Done"
				display(msg, logger)
				
			if numpy.mod(n_updates, options.validFreq) == 0:
				use_noise.set_value(0.)
				
				cost_val = pred_probs(f_log_cost, prepare_data, (valid_x, valid_y, valid_z), kf_valid, options)
				acc_val = pred_acc(f_pred, os.path.join(options.folder, "current_valid_score"), prepare_data, (valid_x, valid_y, valid_z), kf_valid, options)
				err_val = 1.0 - acc_val
				history_errs.append(err_val)
				cost_tst = pred_probs(f_log_cost, prepare_data, (test_x, test_y, test_z), kf_test, options)
				acc_tst = pred_acc(f_pred, os.path.join(options.folder, "current_test_score"), prepare_data, (test_x, test_y, test_z), kf_test, options)
				
				msg = "\nValid cost: %f\n"	\
					  "Valid accuracy %f\n"	\
					  "Test cost: %f\n"		\
					  "Test accuracy: %f\n" 	\
					  "lrate: %f" %(cost_val, acc_val, cost_tst, acc_tst, options.lr)
				display(msg, logger)
				
				valid_acc_record.append(acc_val)
				test_acc_record.append(acc_tst)
				
				if best_p == None or err_val <= numpy.array(history_errs).min():
					best_p = unzip(tparams)
					best_epoch_num = e
					wait_counter = 0
					
				if err_val > numpy.array(history_errs).min():
					wait_counter += 1
					
				if wait_counter >= wait_N:
					msg = "wait_counter max, need to half the lr"
					display(msg, logger)
					bad_counter += 1
					wait_counter = 0
					msg = "bad_counter: %d" %bad_counter
					display(msg, logger)
					options.lr *= 0.5
					lr_change_list.append(e)
					msg = "lrate change to: %f" %(options.lr)
					display(msg, logger)
					zipp(best_p, tparams)
				
				if bad_counter > options.patience:
					estop = True
					break
		
		msg = "Seen %d samples" %n_samples
		display(msg, logger)
		
		if estop:
			msg = "Early Stop!"
			display(msg, logger)
			break
	
	end_time = time.time()
	msg = "Optimizing time: %f seconds" %(end_time - start_time)
	display(msg, logger)
			
	with open(os.path.join(options.folder, 'record.csv'), 'w') as f:
		f.write(str(best_epoch_num) + '\n')
		f.write(','.join(map(str,lr_change_list)) + '\n')
		f.write(','.join(map(str,valid_acc_record)) + '\n')
		f.write(','.join(map(str,test_acc_record)) + '\n')
	
	if best_p is not None:
		zipp(best_p, tparams)
	
	use_noise.set_value(0.)
	
	msg = "\n" + "=" * 80 + "\nFinal Result\n" + "=" * 80
	display(msg, logger)
	
	cost_tra = pred_probs(f_log_cost, prepare_data, (train_x, train_y, train_z), beta_train, options)
	acc_tra = pred_acc(f_pred, os.path.join(options.folder, "train_score"), prepare_data, (train_x, train_y, train_z), beta_train, options)
	cost_val = pred_probs(f_log_cost, prepare_data, (valid_x, valid_y, valid_z), kf_valid, options)
	acc_val = pred_acc(f_pred, os.path.join(options.folder, "valid_score"), prepare_data, (valid_x, valid_y, valid_z), kf_valid, options)
	cost_tst = pred_probs(f_log_cost, prepare_data, (test_x, test_y, test_z), kf_test, options)
	acc_tst = pred_acc(f_pred, os.path.join(options.folder, "test_score"), prepare_data, (test_x, test_y, test_z), kf_test, options)
	msg = "\nTrain cost: %f\n"		\
		  "Train accuracy: %f\n"	\
		  "Valid cost: %f\n"		\
		  "Valid accuracy: %f\n"	\
		  "Test cost: %f\n"			\
		  "Test accuracy: %f\n" 	\
		  "best epoch: %d" %(cost_tra, acc_tra, cost_val, acc_val, cost_tst, acc_tst, best_epoch_num)
	display(msg, logger)
	if best_p is not None:
		params = best_p
	else:
		params = unzip(tparams)
	numpy.savez(os.path.join(options.folder, options.saveto), **params)
	pkl.dump(options, open('%s.pkl' %os.path.join(options.folder, options.saveto), 'wb'))
	msg = "Finished"
	display(msg, logger)
	
def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument("--folder", help = "the dir of model", default = "")
	parser.add_argument("--file_dic", help = "the file of vocabulary", default = "./data/workshop/test_dic.pkl")
	parser.add_argument("--file_train", help = "the file of training data", default = "./data/workshop/train.pkl")
	parser.add_argument("--file_valid", help = "the file of valid data", default = "./data/workshop/valid.pkl")
	parser.add_argument("--file_test", help = "the file of testing data", default = "./data/workshop/test.pkl")
	parser.add_argument("--file_emb", help = "the file of embedding", default = "")
	parser.add_argument("--file_log", help = "the log file", default = "train.log")
	parser.add_argument("--reload_model", help = "the pretrained model", default = "")
	parser.add_argument("--saveto", help = "the file to save the parameter", default = "model")
	parser.add_argument("--dic", help = "word2idx", default = None, type = object)
	
	parser.add_argument("--size_vocab", help = "the size of vocabulary", default = 10000, type = int)
	
	parser.add_argument("--dim_emb", help = "the dimension of the word embedding", default = 300, type = int)
	parser.add_argument("--dim_lstm_input_0", help = "the dimension of the LSTM hidden layer", default = 300, type = int)
	parser.add_argument("--dim_mlp_1", help = "the dimension of the MLP layer", default = 300, type = int)
	parser.add_argument("--dim_lstm_input_1", help = "the dimension of the LSTM hidden layer", default = 300, type = int)
	parser.add_argument("--dim_mlp_2", help = "the dimension of the MLP layer", default = 300, type = int)
	parser.add_argument("--dim_dense", help = "the dimension of the MLP layer", default = 300, type = int)
	parser.add_argument("--num_class", help = "the dimension of the MLP layer", default = 3, type = int)
	
	parser.add_argument("--num_block", help = "the num of blocks", default = 3, type = int)
	
	parser.add_argument("--optimizer", help = "optimization algorithm", default = "adam")
	parser.add_argument("--batch_size", help = "batch size", default = 64, type = int)
	parser.add_argument("--maxlen", help = "max length of sentence", default = 100, type = int)
	parser.add_argument("--seed", help = "the seed of random", default = 345, type = int)
	parser.add_argument("--dispFreq", help = "the frequence of display", default = 100, type = int)
	parser.add_argument("--validFreq", help = "the frequence of valid", default = -1, type = int)
	parser.add_argument("--saveFreq", help = "the frequence of saving", default = -1, type = int)
	parser.add_argument("--nepochs", help = "the max epoch", default = 5000, type = int)
	parser.add_argument("--lr", help = "the initial learning rate", default = 0.0002, type = float)
	parser.add_argument("--dropout_rate", help = "keep rate", default = 0.8, type = float)
	parser.add_argument("--patience", help = "used to early stop", default = 8, type = int)
	parser.add_argument("--decay", help = "the flag to indicate whether to decay the learning rate", action = "store_false", default = True)
	parser.add_argument("--decay_c", help = "decay rate", default = 0, type = float)
	parser.add_argument("--clip_c", help = "grad clip", default = 10, type = float)
	parser.add_argument("--debug", help = "mode flag", action = "store_false", default = True)
	
	options = parser.parse_args(argv)
	train(options)
	
if "__main__" == __name__:
	main(sys.argv[1:])