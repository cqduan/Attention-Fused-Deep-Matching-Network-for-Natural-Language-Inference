import re
import os

import theano
import theano.tensor as tensor
from theano import config
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

import numpy
import math

import copy

def _p(pp, name):
	return '%s_%s' % (pp, name)

def ReLU(x):
    y = tensor.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = tensor.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = tensor.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
	
def rand_weight(dim_in, dim_out):
	W_bound = numpy.sqrt(6. / (dim_in + dim_out))
	W = numpy.random.uniform(low = -W_bound, high = W_bound, size = (dim_in, dim_out)).astype(config.floatX)
	return W

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
	"""
	Random weights drawn from a Gaussian
	"""
	if nout is None:
		nout = nin
	if nout == nin and ortho:
		W = ortho_weight(nin)
	else:
		W = scale * numpy.random.randn(nin, nout)
	return W.astype('float32')
	
def ortho_weight(ndim):
	W = numpy.random.randn(ndim, ndim)
	u, s, v = numpy.linalg.svd(W)
	return u.astype(config.floatX)

def numpy_floatX(data):
	return numpy.asarray(data, dtype=config.floatX)

def reverse(tensor):
	return tensor[::-1]

class MLP_layer:
	def __init__(self, prefix = "MLP", dim_in = None, dim_out = None):
		self.prefix = prefix
		self.dim_in = dim_in
		self.dim_out = dim_out
		
	def init_params(self, options, params):
		if self.dim_in == None:
			self.dim_in = options.MLP_in
		if self.dim_out == None:
			self.dim_out = options.MLP_out
			
		W = norm_weight(self.dim_in, self.dim_out)
		params[_p(self.prefix, "W")] = W
		
		b = numpy.zeros(self.dim_out, dtype = config.floatX)
		params[_p(self.prefix, "b")] = b
		
	def build(self, tparams, state_below, options):
		return tensor.dot(state_below, tparams[_p(self.prefix, "W")]) + tparams[_p(self.prefix, "b")]

class LSTM_layer:
	def __init__(self, prefix = "lstm", dim_in = None, dim_out = None):
		self.prefix = prefix
		self.dim_in = dim_in
		self.dim_out = dim_out
		
	def init_params(self, options, params):
		if self.dim_in == None:
			self.dim_in = options.dim_emb
		if self.dim_out == None:
			self.dim_out = options.dim_lstm
			
		W = numpy.concatenate([norm_weight(self.dim_in, self.dim_out),
							   norm_weight(self.dim_in, self.dim_out),
							   norm_weight(self.dim_in, self.dim_out),
							   norm_weight(self.dim_in, self.dim_out)],
							   axis = 1)
		params[_p(self.prefix, "W")] = W
		
		U = numpy.concatenate([ortho_weight(self.dim_out),
							   ortho_weight(self.dim_out),
							   ortho_weight(self.dim_out),
							   ortho_weight(self.dim_out)],
							   axis = 1)
		params[_p(self.prefix, "U")] = U
		
		b = numpy.zeros((4 * self.dim_out,), dtype=config.floatX)
		params[_p(self.prefix, "b")] = b
		
		return params
	
	def build(self, tparams, state_below, options, mask=None):
		nsteps = state_below.shape[0]
		if state_below.ndim == 3:
			n_samples = state_below.shape[1]
		else:
			n_samples = 1

		assert mask is not None
		
		def _slice(_x, n, dim):
			if _x.ndim == 3:
				return _x[:, :, n * dim:(n + 1) * dim]
			return _x[:, n * dim:(n + 1) * dim]
		
		def _step(m_, x_, h_, c_):
			preact = tensor.dot(h_, tparams[_p(self.prefix, 'U')])
			preact += x_

			i = tensor.nnet.sigmoid(_slice(preact, 0, self.dim_out))
			f = tensor.nnet.sigmoid(_slice(preact, 1, self.dim_out))
			o = tensor.nnet.sigmoid(_slice(preact, 2, self.dim_out))
			c = tensor.tanh(_slice(preact, 3, self.dim_out))

			c = f * c_ + i * c
			c = m_[:, None] * c + (1. - m_)[:, None] * c_
			#c = m_[:, None] * c

			h = o * tensor.tanh(c)
			h = m_[:, None] * h + (1. - m_)[:, None] * h_

			return h, c
		
		state_below = (tensor.dot(state_below, tparams[_p(self.prefix, 'W')]) + tparams[_p(self.prefix, 'b')])
		
		rval, _ = theano.scan(_step,
							  sequences = [mask, state_below],
							  outputs_info = [tensor.alloc(numpy_floatX(0.), n_samples, self.dim_out),
											  tensor.alloc(numpy_floatX(0.), n_samples, self.dim_out)],
							  name = _p(self.prefix, 'layers'),
							  n_steps = nsteps)
		
		out_val = rval[0]
		
		return out_val
	
class EMB_layer:
	def __init__(self, prefix = "", dim_in = None, dim_out = None):
		self.prefix = prefix
		self.dim_in = dim_in
		self.dim_out = dim_out
	
	def init_params(self, options, params):
		if self.dim_in == None:
			self.dim_in = options.size_vocab
		if self.dim_out == None:
			self.dim_out = options.dim_emb
		self.dic = options.dic
		
		randn = norm_weight(self.dim_in, self.dim_out)
		params[_p(self.prefix, 'Wemb')] = randn.astype(config.floatX)
		
		if options.file_emb:
			self.load_emb(options.file_emb, params)
		return params
	
	def load_emb(self, textFile, params):
		print 'load emb from ' + textFile
		filein = open(textFile, 'r')
		emb_dict = {}
		emb_p = re.compile(r" |\t")
		for line in filein:
			array = emb_p.split(line.strip())
			vector = [float(array[i]) for i in range(1, len(array))]
			word = array[0]
			emb_dict[word] = vector
		filein.close()
		print "find %d words in %s" %(len(emb_dict), textFile)
		
		count = 0
		for k, v in self.dic.items():
			if k in emb_dict:
				params[_p(self.prefix, 'Wemb')][v] = copy.deepcopy(emb_dict[k])
				count += 1
		print "Summary:\n\ttotal: %d\n\tappear: %d" %(len(self.dic), count)
	
	def build(self, tparams, options):
		#L X batch X win
		x = tensor.matrix(dtype = "int64")
		mask_x = tensor.matrix(dtype = config.floatX)
		
		y = tensor.matrix(dtype = "int64")
		mask_y = tensor.matrix(dtype = config.floatX)
		
		z = tensor.vector(dtype = "int64")
		
		#L X batch X emb
		emb_x = tparams[_p(self.prefix, "Wemb")][x.flatten()].reshape([x.shape[0], x.shape[1], options.dim_emb])
		emb_y = tparams[_p(self.prefix, "Wemb")][y.flatten()].reshape([y.shape[0], y.shape[1], options.dim_emb])
		
		return x, mask_x, emb_x, y, mask_y, emb_y, z
		
class Concat_layer:
	def __init__(self, prefix='concat'):
		self.prefix = prefix
    
	def build(self, input1, input2, axis=-1):
		self.out = tensor.concatenate([input1, input2], axis)
		return self.out

	def output(self):
		return self.out

def dropout_layer(state_before, use_noise, trng, options):
	proj = tensor.switch(use_noise,
			      (state_before * trng.binomial(state_before.shape, p=options.dropout_rate, n=1, dtype=state_before.dtype)),
				state_before * options.dropout_rate)
	return proj
	
def grad_norm(grads):
	grad_norm = 0.0
	for g in grads:
		grad_norm += (numpy.asarray(g) ** 2).sum()
		
	return numpy.sqrt(grad_norm)
