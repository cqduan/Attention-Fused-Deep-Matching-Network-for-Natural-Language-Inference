import os
import numpy
import theano
import cPickle
import subprocess

def datafold(textFile):
	x, y, z = cPickle.load(open(textFile, "rb"))
	return x, y, z
	
def dicfold(textFile):
	word2idx, label2idx = cPickle.load(open(textFile, "rb"))
	return word2idx, label2idx

def get_model_size(params):
	total_size = 0
	for k, v in params.items():
		total_size += v.size
	return total_size

def get_minibatches_idx(n, minibatch_size, shuffle=False):
	"""
	Used to shuffle the dataset at each iteration.
	"""

	idx_list = numpy.arange(n, dtype="int64")

	if shuffle:
		numpy.random.shuffle(idx_list)

	minibatches = []
	minibatch_start = 0
	for i in range(n // minibatch_size):
		minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
		minibatch_start += minibatch_size

	if (minibatch_start != n):
		# Make a minibatch out of what is left
		minibatches.append(idx_list[minibatch_start:])

	return zip(range(len(minibatches)), minibatches)

def split_idx(nsentences, valid_rate):
	idx_list = numpy.arange(nsentences, dtype="int64")
	numpy.random.shuffle(idx_list)
	return idx_list[:int(nsentences * valid_rate)], idx_list[int(nsentences * valid_rate):]
	
def prepare_data(seqs, maxlen = None):
	"""Create the matrices from the datasets.

	This pad each sequence to the same lenght: the lenght of the
	longuest sequence or maxlen.

	if maxlen is set, we will cut all sequence to this maximum
	lenght.

	This swap the axis!
	"""
	lengths = [len(s) for s in seqs]
	
	if maxlen:
		new_seqs = []
		new_lengths = []
		for l, s in zip(lengths, seqs):
			if l < maxlen:
				new_seqs.append(s)
				new_lengths.append(l)
			else:
				new_seqs.append(s[: maxlen])
				new_lengths.append(maxlen)
		seqs = new_seqs
		lengths = new_lengths
		
		if len(lengths) < 1:
			return None, None
	
	n_samples = len(seqs)
	max_len = max(lengths)
	
	x = numpy.zeros((max_len, n_samples)).astype('int64')
	x_mask = numpy.zeros((max_len, n_samples)).astype(theano.config.floatX)
	
	for idx, s in enumerate(seqs):
		x[:lengths[idx], idx] = s
		x_mask[:lengths[idx], idx] = 1.

	return x, x_mask
	
def getA(answers, results):
	total = 0.
	right = 0.
	for i in range(len(answers)):
		if answers[i] == results[i]:
			right += 1
		total += 1
	return right / total	

def getP(answers, results, tag):
	total = 0.
	right = 0.
	for i in range(len(results)):
		if results[i] == tag:
			total += 1
		if answers[i] == tag and results[i] == tag:
			right += 1
	#print "P total: %d" %total
	#print "P right: %d" %right
	return 0. if total == 0 else right / total

def getR(answers, results, tag):
	total = 0.
	right = 0.
	for i in range(len(answers)):
		if answers[i] == tag:
			total += 1
		if answers[i] == tag and results[i] == tag:
			right += 1
	#print "R total: %d" %total
	#print "R right: %d" %right
	return 0. if total == 0 else right / total

def getF1(precision, recall):
	return 0. if (precision + recall) == 0 else 2. * precision * recall / (precision + recall)

def evaluate(answers, results, filename):
	#print "answers.size: %d" %len(answers)
	#print "results.size: %d" %len(results)
	f = open(filename, "w")
	for p in results:
		f.write(str(p) + "\n")
	f.close()
	
	pos_P = getP(answers, results, 1)
	pos_R = getR(answers, results, 1)
	pos_F1 = getF1(pos_P, pos_R)

	neg_P = getP(answers, results, 0)
	neg_R = getR(answers, results, 0)
	neg_F1 = getF1(neg_P, neg_R)

	accuracy = getA(answers, results)
	
	return pos_P, pos_R, pos_F1, neg_P, neg_R, neg_F1, accuracy
	
def save_score(score, filename):
	with open(filename, "w") as f:
		for s in score:
			f.write("%f\n" %s)

def save_label(label, filename):
	with open(filename, "w") as f:
		for l in label:
			f.write("%d\n" %l)
			
def pred_acc(f_pred, pred_file, prepare_data, data, kf_data, options):
	data_acc = 0
	n_done = 0
	
	labels = []
	preds = []
	for _, data_index in kf_data:
		x = [data[0][t] for t in data_index]
		y = [data[1][t] for t in data_index]
		z = [data[2][t] for t in data_index]
		
		x_, mask_x_ = prepare_data(x)
		y_, mask_y_ = prepare_data(y)
		z_ = numpy.array(z)
		
		labels.extend(z_)
		preds.extend(f_pred(x_, y_, mask_x_, mask_y_))
	data_acc = getA(labels, preds)
	
	with open(pred_file, "w") as f:
		for pred in preds:
			f.write("%d\n" %pred)
	
	return data_acc

def pred_probs(f_log_prob, prepare_data, data, data_kf, options):
	probs = []
	
	for _, data_index in data_kf:
		x = [data[0][t] for t in data_index]
		y = [data[1][t] for t in data_index]
		z = [data[2][t] for t in data_index]
		
		x_, mask_x_ = prepare_data(x)
		y_, mask_y_ = prepare_data(y)
		z_ = numpy.array(z)
		
		pprobs = f_log_prob(x_, y_, mask_x_, mask_y_, z_)
		probs.append(pprobs)
	
	return numpy.array(probs).mean()
