import sys
import os
import codecs
import argparse
import logging
import pickle as pkl

label2idx = {"entailment": 0, "neutral": 1, "contradiction": 2}

def display(msg, logger):
	print(msg)
	logger.info(msg)
	
def load_data(textFile, logger, lower_case = False):
	msg = "loading data from %s" %textFile
	display(msg, logger)
	
	pres = []
	hyps = []
	rels = []
	with codecs.open(textFile, "r", encoding = "utf8") as f:
		f.readline()
		for line in f:
			if lower_case:
				sents = line.strip().lower().split("\t")
			else:
				sents = line.strip().split("\t")
			
			if sents[0] == "-":
				continue
			
			x = " ".join([word for word in sents[1].strip().split(" ") if word not in ("(", ")")])
			pres.append(x)
			y = " ".join([word for word in sents[2].strip().split(" ") if word not in ("(", ")")])
			hyps.append(y)
			z = label2idx[sents[0].strip()]
			rels.append(z)
	
	msg = "find %d instances in %s" %(len(rels), textFile)
	display(msg, logger)
	
	return pres, hyps, rels

def build_dic(sents, logger):
	msg = "build dictionary..."
	display(msg, logger)
	
	vocab = {}
	for sent in sents:
		for word in sent.split():
			if word in vocab:
				vocab[word] += 1
			else:
				vocab[word] = 1
	
	vocab = [pair[0] for pair in sorted(vocab.items(), key = lambda x: x[1], reverse = True)]
	
	word2idx = {}
	word2idx["<PAD>"] = 0
	word2idx["<UNK>"] = 1
	word2idx["<BOS>"] = 2
	word2idx["<EOS>"] = 3
	
	for idx, word in enumerate(vocab):
		word2idx[word] = idx + 4
	
	msg = "#vocab: %d" %(len(word2idx))
	display(msg, logger)
	
	return word2idx
	
def str2list(sent, options):
	if options.if_eos_bos:
		new_sent = ["<BOS>"] + sent.split(" ") + ["<EOS>"]
	else:
		new_sent = sent.split(" ")
	
	return new_sent
	
def sent2idx(sents, word2idx, data_name, logger, options):
	msg = "transfer sentences in %s from word to idx" %data_name
	display(msg, logger)
	sent_idxs = [[word2idx[word] if word in word2idx else word2idx["<UNK>"] for word in str2list(sent, options)] for sent in sents]
	if options.num_word > 0:
		sent_idxs = [[word if word < options.num_word else word2idx["<UNK>"] for word in sent] for sent in sent_idxs]
	lens = [len(sent) for sent in sent_idxs]
	msg = "%s lengths: %d(min), %d(max)" %(data_name, min(lens), max(lens))
	display(msg, logger)
	return sent_idxs
	
def preprocess(options):
	if not os.path.exists(options.folder):
		os.mkdir(options.folder)
	
	logging.basicConfig(level = logging.DEBUG,
						format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s",
						filename = os.path.join(options.folder, options.file_log),
						filemode = "w")
	logger = logging.getLogger()
	
	logger.info("python %s" %(" ".join(sys.argv)))
	
	train_x, train_y, train_z = load_data(options.in_file_train, logger, options.lower_case)
	dev_x, dev_y, dev_z = load_data(options.in_file_dev, logger, options.lower_case)
	test_x, test_y, test_z = load_data(options.in_file_test, logger, options.lower_case)
	
	word2idx = build_dic(train_x + train_y + dev_x + dev_y + test_x + test_y, logger)
	#word2idx = build_dic(train_x + train_y, logger)
	
	train_x_idx = sent2idx(train_x, word2idx, "train_x", logger, options)
	train_y_idx = sent2idx(train_y, word2idx, "train_y", logger, options)
	
	dev_x_idx = sent2idx(dev_x, word2idx, "dev_x", logger, options)
	dev_y_idx = sent2idx(dev_y, word2idx, "dev_y", logger, options)
	
	test_x_idx = sent2idx(test_x, word2idx, "test_x", logger, options)
	test_y_idx = sent2idx(test_y, word2idx, "test_y", logger, options)
	
	train = (train_x_idx, train_y_idx, train_z)
	dev = (dev_x_idx, dev_y_idx, dev_z)
	test = (test_x_idx, test_y_idx, test_z)	
	dics = (word2idx, label2idx)
	
	msg = "saving training data into %s" %os.path.join(options.folder, options.out_file_train)
	display(msg, logger)
	pkl.dump(train, open(os.path.join(options.folder, options.out_file_train), "wb"))
	msg = "saving dev data into %s" %os.path.join(options.folder, options.out_file_dev)
	display(msg, logger)
	pkl.dump(dev, open(os.path.join(options.folder, options.out_file_dev), "wb"))
	msg = "saving test data into %s" %os.path.join(options.folder, options.out_file_test)
	display(msg, logger)
	pkl.dump(test, open(os.path.join(options.folder, options.out_file_test), "wb"))
	msg = "saving dictionary into %s" %os.path.join(options.folder, options.out_file_dic)
	display(msg, logger)
	pkl.dump(dics, open(os.path.join(options.folder, options.out_file_dic), "wb"))
	
def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument("--folder", help = "the dir of model", default = "workshop")
	parser.add_argument("--in_file_train", help = "the training file", default = "snli_1.0/snli_1.0_train.txt")
	parser.add_argument("--in_file_dev", help = "the dev file", default = "snli_1.0/snli_1.0_dev.txt")
	parser.add_argument("--in_file_test", help = "the test file", default = "snli_1.0/snli_1.0_test.txt")
	parser.add_argument("--out_file_train", help = "the training file", default = "snli_1.0_train.pkl")
	parser.add_argument("--out_file_dev", help = "the dev file", default = "snli_1.0_dev.pkl")
	parser.add_argument("--out_file_test", help = "the test file", default = "snli_1.0_test.pkl")
	parser.add_argument("--out_file_dic", help = "the dictionary file", default = "snli_1.0_dic.pkl")
	parser.add_argument("--file_log", help = "the log file", default = "log.txt")
	parser.add_argument("--lower_case", help = "if true, lowercase the data", action = "store_false", default = True)
	parser.add_argument("--if_eos_bos", help = "if true, the <EOS> and <BOS> will be instered into the sentence", action = "store_false", default = True)
	parser.add_argument("--num_word", help = "the max size of vocab", default = 100000)
	options = parser.parse_args(argv)
	preprocess(options)
	
if "__main__" == __name__:
	main(sys.argv[1:])
