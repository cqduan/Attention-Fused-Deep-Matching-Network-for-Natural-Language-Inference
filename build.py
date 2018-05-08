from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import sys
sys.path.append(".")
from lib import *

optimizer = {"adam": adam, "adadelta": adadelta}

def _p(pp, name):
	return '%s_%s' % (pp, name)

def zipp(params, tparams):
	for kk, vv in params.iteritems():
		tparams[kk].set_value(vv)

def unzip(zipped):
	new_params = OrderedDict()
	for kk, vv in zipped.iteritems():
		new_params[kk] = vv.get_value()
	return new_params

def load_params(path, params):
	pp = numpy.load(path)
	for kk, vv in params.iteritems():
		if kk not in pp:
			raise Warning("%s is not in the archive" % kk)
		params[kk] = pp[kk]

	return params

def init_tparams(params):
	tparams = OrderedDict()
	for kk, pp in params.iteritems():
		tparams[kk] = theano.shared(params[kk], name=kk)
	return tparams
	
def build_model(options):
	trng = RandomStreams(options.seed)
	
	# Used for dropout.
	use_noise = theano.shared(numpy_floatX(0.))

	params = OrderedDict()
	
	emb_layer = EMB_layer()
	emb_layer.init_params(options, params)
	
	lstm_fw_input_0 = LSTM_layer(prefix = "lstm_fw_input_0", dim_in = options.dim_emb, dim_out = options.dim_lstm_input_0)
	lstm_fw_input_0.init_params(options, params)
	
	lstm_bw_input_0 = LSTM_layer(prefix = "lstm_bw_input_0", dim_in = options.dim_emb, dim_out = options.dim_lstm_input_0)
	lstm_bw_input_0.init_params(options, params)
	
	for i in range(options.num_block):
		external_att_W = norm_weight(options.dim_lstm_input_0 * 2, options.dim_lstm_input_0 * 2)
		params[_p("external_att_W", str(i))] = external_att_W
	
		external_att_W_x = norm_weight(1, options.dim_lstm_input_0 * 2).flatten()
		params[_p("external_att_W_x", str(i))] = external_att_W_x
	
		external_att_W_y = norm_weight(1, options.dim_lstm_input_0 * 2).flatten()
		params[_p("external_att_W_y", str(i))] = external_att_W_y
	
	mlp_layer_1s = []
	lstm_fw_input_1s = []
	lstm_bw_input_1s = []
	mlp_layer_2s = []
	lstm_fw_input_2s = []
	lstm_bw_input_2s = []
	for i in range(options.num_block):
		mlp_layer_1 = MLP_layer(prefix = _p("mlp_layer_1", str(i)), dim_in = options.dim_lstm_input_0 * 2 * 4, dim_out = options.dim_mlp_1)
		mlp_layer_1.init_params(options, params)
		mlp_layer_1s.append(mlp_layer_1)
	
		lstm_fw_input_1 = LSTM_layer(prefix = _p("lstm_fw_input_1", str(i)), dim_in = options.dim_mlp_1, dim_out = options.dim_lstm_input_1)
		lstm_fw_input_1.init_params(options, params)
		lstm_fw_input_1s.append(lstm_fw_input_1)
		
		lstm_bw_input_1 = LSTM_layer(prefix = _p("lstm_bw_input_1", str(i)), dim_in = options.dim_mlp_1, dim_out = options.dim_lstm_input_1)
		lstm_bw_input_1.init_params(options, params)
		lstm_bw_input_1s.append(lstm_bw_input_1)
		
		mlp_layer_2 = MLP_layer(prefix = _p("mlp_layer_2", str(i)), dim_in = options.dim_lstm_input_1 * 2 * 4, dim_out = options.dim_mlp_2)
		mlp_layer_2.init_params(options, params)
		mlp_layer_2s.append(mlp_layer_2)
		
		lstm_fw_input_2 = LSTM_layer(prefix = _p("lstm_fw_input_2", str(i)), dim_in = options.dim_mlp_2, dim_out = options.dim_lstm_input_0)
		lstm_fw_input_2.init_params(options, params)
		lstm_fw_input_2s.append(lstm_fw_input_2)
		
		lstm_bw_input_2 = LSTM_layer(prefix = _p("lstm_bw_input_2", str(i)), dim_in = options.dim_mlp_2, dim_out = options.dim_lstm_input_0)
		lstm_bw_input_2.init_params(options, params)
		lstm_bw_input_2s.append(lstm_bw_input_2)
	
	mlp_layer_3 = MLP_layer(prefix = "dense", dim_in = options.dim_lstm_input_0 * 2 * 4, dim_out = options.dim_dense)
	mlp_layer_3.init_params(options, params)
	
	mlp_layer_4 = MLP_layer(prefix = "predicator", dim_in = options.dim_dense, dim_out = options.num_class)
	mlp_layer_4.init_params(options, params)
	
        for k, v in params.items():
                print k, v.shape

	if options.reload_model:
		load_params(os.path.join(options.folder, options.reload_model), params)
	
	tparams = init_tparams(params)
	
	x, mask_x, emb_x, y, mask_y, emb_y, z = emb_layer.build(tparams, options)
	
	emb_x_dp = dropout_layer(emb_x, use_noise, trng, options)
	emb_y_dp = dropout_layer(emb_y, use_noise, trng, options)
	
	input_fw_x_0 = lstm_fw_input_0.build(tparams, emb_x_dp, options, mask_x)
	input_bw_x_0 = reverse(lstm_bw_input_0.build(tparams, reverse(emb_x_dp), options, reverse(mask_x)))

	input_x_0 = tensor.concatenate([input_fw_x_0, input_bw_x_0], axis = -1) * mask_x[:, :, None]
	
	input_fw_y_0 = lstm_fw_input_0.build(tparams, emb_y_dp, options, mask_y)
	input_bw_y_0 = reverse(lstm_bw_input_0.build(tparams, reverse(emb_y_dp), options, reverse(mask_y)))
	
	input_y_0 = tensor.concatenate([input_fw_y_0, input_bw_y_0], axis = -1) * mask_y[:, :, None]
	
	input_xs = [input_x_0]
	input_ys = [input_y_0]
	for i in range(options.num_block):
		input_x_0 = input_xs[-1]
		input_y_0 = input_ys[-1]
		
		external_weight_matrix = tensor.dot(input_x_0.dimshuffle(1, 0, 2), tparams[_p("external_att_W", str(i))])
		external_weight_matrix = tensor.batched_dot(external_weight_matrix, input_y_0.dimshuffle(1, 2, 0))
		external_weight_vec_x = tensor.dot(input_x_0, tparams[_p("external_att_W_x", str(i))]).dimshuffle(1, 0)
		external_weight_vec_y = tensor.dot(input_y_0, tparams[_p("external_att_W_y", str(i))]).dimshuffle(1, 0)
		external_weight_matrix = external_weight_matrix + external_weight_vec_x[:, :, None] + external_weight_vec_y[:, None, :]
		external_weight_matrix_x = tensor.exp(external_weight_matrix - external_weight_matrix.max(axis = 1, keepdims = True)).dimshuffle(1, 2, 0)
		external_weight_matrix_y = tensor.exp(external_weight_matrix - external_weight_matrix.max(axis = 2, keepdims = True)).dimshuffle(1, 2, 0)
		external_weight_matrix_x = external_weight_matrix_x * mask_x[:, None, :]
		external_weight_matrix_y = external_weight_matrix_y * mask_y[None, :, :]
		external_alpha = external_weight_matrix_x / external_weight_matrix_x.sum(axis = 0, keepdims = True)
		external_beta = external_weight_matrix_y / external_weight_matrix_y.sum(axis = 1, keepdims = True)
	
		input_y_external_att = (input_x_0.dimshuffle(0, 'x', 1, 2) * external_alpha.dimshuffle(0, 1, 2, 'x')).sum(0)
		input_x_external_att = (input_y_0.dimshuffle('x', 0, 1, 2) * external_beta.dimshuffle(0, 1, 2, 'x')).sum(1)
		
		input_x_0_cat = tensor.concatenate([input_x_0, input_x_external_att, input_x_0 - input_x_external_att, input_x_0 * input_x_external_att], axis = -1)
		input_y_0_cat = tensor.concatenate([input_y_0, input_y_external_att, input_y_0 - input_y_external_att, input_y_0 * input_y_external_att], axis = -1)

		input_x_0_cat_dp = dropout_layer(input_x_0_cat, use_noise, trng, options)
		input_y_0_cat_dp = dropout_layer(input_y_0_cat, use_noise, trng, options)

		input_x_mlp_1 = tensor.nnet.relu(mlp_layer_1s[i].build(tparams, input_x_0_cat_dp, options))
		input_y_mlp_1 = tensor.nnet.relu(mlp_layer_1s[i].build(tparams, input_y_0_cat_dp, options))
		
		input_x_mlp_1_dp = dropout_layer(input_x_mlp_1, use_noise, trng, options)
		input_y_mlp_1_dp = dropout_layer(input_y_mlp_1, use_noise, trng, options)
		
		input_fw_x_1 = lstm_fw_input_1s[i].build(tparams, input_x_mlp_1_dp, options, mask_x)
		input_bw_x_1 = reverse(lstm_bw_input_1s[i].build(tparams, reverse(input_x_mlp_1_dp), options, reverse(mask_x)))
		
		input_x_1 = tensor.concatenate([input_fw_x_1, input_bw_x_1], axis = -1) * mask_x[:, :, None]
		input_x_1 = input_x_0 + input_x_1
		
		input_fw_y_1 = lstm_fw_input_1s[i].build(tparams, input_y_mlp_1_dp, options, mask_y)
		input_bw_y_1 = reverse(lstm_bw_input_1s[i].build(tparams, reverse(input_y_mlp_1_dp), options, reverse(mask_y)))
		
		input_y_1 = tensor.concatenate([input_fw_y_1, input_bw_y_1], axis = -1) * mask_y[:, :, None]
		input_y_1 = input_y_0 + input_y_1
		
		internal_weight_matrix_x = tensor.batched_dot(input_x_1.dimshuffle(1, 0, 2), input_x_1.dimshuffle(1, 2, 0))
		internal_weight_matrix_x = tensor.exp(internal_weight_matrix_x - internal_weight_matrix_x.max(axis = 1, keepdims = True)).dimshuffle(1, 2, 0)
		internal_weight_matrix_x = internal_weight_matrix_x * mask_x[:, None, :]
		internal_alpha = internal_weight_matrix_x / internal_weight_matrix_x.sum(axis = 0, keepdims = True)
		
		internal_weight_matrix_y = tensor.batched_dot(input_y_1.dimshuffle(1, 0, 2), input_y_1.dimshuffle(1, 2, 0))
		internal_weight_matrix_y = tensor.exp(internal_weight_matrix_y - internal_weight_matrix_y.max(axis = 1, keepdims = True)).dimshuffle(1, 2, 0)
		internal_weight_matrix_y = internal_weight_matrix_y * mask_y[:, None, :]
		internal_beta = internal_weight_matrix_y / internal_weight_matrix_y.sum(axis = 0, keepdims = True)
		
		input_x_internal_att = (input_x_1.dimshuffle(0, 'x', 1, 2) * internal_alpha.dimshuffle(0, 1, 2, 'x')).sum(0)
		input_y_internal_att = (input_y_1.dimshuffle(0, 'x', 1, 2) * internal_beta.dimshuffle(0, 1, 2, 'x')).sum(0)
		
		input_x_1_cat = tensor.concatenate([input_x_1, input_x_internal_att, input_x_1 - input_x_internal_att, input_x_1 * input_x_internal_att], axis = -1)
		input_y_1_cat = tensor.concatenate([input_y_1, input_y_internal_att, input_y_1 - input_y_internal_att, input_y_1 * input_y_internal_att], axis = -1)

		input_x_1_cat_dp = dropout_layer(input_x_1_cat, use_noise, trng, options)
		input_y_1_cat_dp = dropout_layer(input_y_1_cat, use_noise, trng, options)

		input_x_mlp_2 = tensor.nnet.relu(mlp_layer_2s[i].build(tparams, input_x_1_cat_dp, options))
		input_y_mlp_2 = tensor.nnet.relu(mlp_layer_2s[i].build(tparams, input_y_1_cat_dp, options))
		
		input_x_mlp_2_dp = dropout_layer(input_x_mlp_2, use_noise, trng, options)
		input_y_mlp_2_dp = dropout_layer(input_y_mlp_2, use_noise, trng, options)
		
		input_fw_x_2 = lstm_fw_input_2s[i].build(tparams, input_x_mlp_2_dp, options, mask_x)
		input_bw_x_2 = reverse(lstm_bw_input_2s[i].build(tparams, reverse(input_x_mlp_2_dp), options, reverse(mask_x)))
		
		input_x_2 = tensor.concatenate([input_fw_x_2, input_bw_x_2], axis = -1) * mask_x[:, :, None]
		input_x_2 = input_x_1 + input_x_2
		
		input_fw_y_2 = lstm_fw_input_2s[i].build(tparams, input_y_mlp_2_dp, options, mask_y)
		input_bw_y_2 = reverse(lstm_bw_input_2s[i].build(tparams, reverse(input_y_mlp_2_dp), options, reverse(mask_y)))
		
		input_y_2 = tensor.concatenate([input_fw_y_2, input_bw_y_2], axis = -1) * mask_y[:, :, None]
		input_y_2 = input_y_1 + input_y_2
		
		input_xs.append(input_x_2)
		input_ys.append(input_y_2)

	vec_x_mean = (input_xs[-1] * mask_x[:, :, None]).sum(axis = 0) / mask_x.sum(axis = 0)[:, None]
	vec_x_max = (input_xs[-1] * mask_x[:, :, None]).max(axis = 0)
	
	vec_y_mean = (input_ys[-1] * mask_y[:, :, None]).sum(axis = 0) / mask_y.sum(axis = 0)[:, None]
	vec_y_max = (input_ys[-1] * mask_y[:, :, None]).max(axis = 0)
	
	vec_x_y = tensor.concatenate([vec_x_mean, vec_x_max, vec_y_mean, vec_y_max], axis = -1)
	
	vec_x_y_dp = dropout_layer(vec_x_y, use_noise, trng, options)
	
	vec_mlp = tensor.tanh(mlp_layer_3.build(tparams, vec_x_y_dp, options))
	
	vec_mlp_dp = dropout_layer(vec_mlp, use_noise, trng, options)
	
	pred_prob = tensor.nnet.nnet.softmax(mlp_layer_4.build(tparams, vec_mlp_dp, options))
	pred = pred_prob.argmax(axis = -1)
	
	off = 1e-8
	if pred_prob.dtype == 'float16':
		off = 1e-6
	
	log_cost = -tensor.log(pred_prob[tensor.arange(x.shape[1]), z] + off).mean()
	cost = log_cost
	if options.decay_c > 0.:
		decay_c = theano.shared(numpy.float32(options.decay_c), name='decay_c')
		weight_decay = 0.
		for kk, vv in tparams.iteritems():
			weight_decay += (vv ** 2).sum()
		weight_decay *= decay_c
		cost += weight_decay
	
	grads = tensor.grad(cost, wrt = tparams.values())
	g2 = 0.
	for g in grads:
		g2 += (g ** 2).sum()
	grad_norm = tensor.sqrt(g2)
	
	if options.clip_c > 0.:
		new_grads = []
		for g in grads:
			new_grads.append(tensor.switch(g2 > options.clip_c ** 2, g * options.clip_c / tensor.sqrt(g2), g))
		grads = new_grads
		
	return params, tparams, use_noise, x, mask_x, emb_x, y, mask_y, emb_y, z, pred_prob, pred, log_cost, cost, grad_norm, grads
	
def build_optimizer(lr, tparams, x, y, mask_x, mask_y, z, pred_prob, pred, log_cost, cost, grad_norm, grads, options):
	f_grad_shared, f_update = optimizer[options.optimizer](lr, tparams, (x, y, mask_x, mask_y, z), cost, grads)
	f_log_cost = theano.function(inputs = [x, y, mask_x, mask_y, z], outputs = log_cost, name = "f_log_cost")
	f_grad_norm = theano.function(inputs = [x, y, mask_x, mask_y, z], outputs = grad_norm, name = "f_grad_norm")
	f_pred_prob = theano.function(inputs = [x, y, mask_x, mask_y], outputs = pred_prob, name = "f_pred_prob")
	f_pred = theano.function(inputs = [x, y, mask_x, mask_y], outputs = pred, name = "f_pred")
	return f_grad_shared, f_update, f_log_cost, f_grad_norm, f_pred_prob, f_pred
