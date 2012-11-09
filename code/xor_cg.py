# feedforward neural network
# input(2) -> hidden(2) -> output(1)
# trained on the XOR problem
# weights optimized using scipy.optimize.fmin_cg with
# gradients computed using backpropagation
# Paul Gribble, November 2012
# paul [at] gribblelab [dot] org

# ipython --pylab

##########################
# some utility functions #
##########################

import time
random.seed(int(time.time()))

# sigmoid activation function
def tansig(x):
	return tanh(x)

# derivative of sigmoid function
# (needed for calculating error gradients using backprop)
def dtansig(x):
	# in case x is a vector, multiply() will do element-wise multiplication
	return 1.0 - (multiply(x,x))

# a function that will compute activations of a layer
def layer_forward(act, wgt):
	return tansig( act * wgt )

# a function to compute errors for a layer
def layer_errors(deltas, wgt):
	return deltas * transpose(wgt)

# a function that will compute error gradients for a layer
def layer_deltas(err, act):
	return multiply(dtansig(act), err) # element-wise multiply

# a function to update weights
# N is learning rate parameter and M is momentum parameter
def layer_update_weights(wgt, deltas, act, wgt_prev_change):
	return transpose(act)*deltas

def net_acts(wgt_hid, wgt_out, pat_in):
	# forward pass
	act_inp = pat_in						   # select the first training example
	act_hid = layer_forward(act_inp, wgt_hid)  # hidden unit activations
	act_out = layer_forward(act_hid, wgt_out)  # output unit activations
	return act_inp, act_hid, act_out

def net_forward(x, params):
	n_in = params[0]
	n_hid = params[1]
	n_out = params[2]
	pat_in = params[3]
	pat_out = params[4]
	n_pat = shape(pat_in)[0]
	wgt_hid = reshape(x[0:n_in*n_hid], (n_in,n_hid))
	wgt_out = reshape(x[n_in*n_hid:], (n_hid,n_out))
	net_out = zeros((n_pat,n_out))
	for j in range(n_pat):
		act_inp,act_hid,act_out = net_acts(wgt_hid, wgt_out, pat_in[j,:])
		net_out[j,:] = act_out
	return net_out

def f(x,params):
	n_in = params[0]
	n_hid = params[1]
	n_out = params[2]
	pat_in = params[3]
	pat_out = params[4]
	n_pat = shape(pat_in)[0]
	wgt_hid = reshape(x[0:n_in*n_hid], (n_in,n_hid))
	wgt_out = reshape(x[n_in*n_hid:], (n_hid,n_out))
	cost = 0.0
	for i in range(n_pat):
		act_inp,act_hid,act_out = net_acts(wgt_hid, wgt_out, pat_in)
		cost += 0.5*sum(square(pat_out - act_out))
	print cost
	return cost

def fd(x,params):
	n_in = params[0]
	n_hid = params[1]
	n_out = params[2]
	pat_in = params[3]
	pat_out = params[4]
	n_pat = shape(pat_in)[0]
	wgt_hid = reshape(x[0:n_in*n_hid], (n_in,n_hid))
	wgt_out = reshape(x[n_in*n_hid:], (n_hid,n_out))
	g = zeros(shape(x))
	for j in range(n_pat):
		act_inp,act_hid,act_out = net_acts(wgt_hid, wgt_out, pat_in)
		err_out = pat_out - act_out
		err_sse = 0.5*sum(square(err_out))
		deltas_out = layer_deltas(err_out, act_out)
		err_hid = layer_errors(deltas_out, wgt_out)
		deltas_hid = layer_deltas(err_hid, act_hid)
		grad_hid = transpose(act_inp)*deltas_hid
		grad_out = transpose(act_hid)*deltas_out
		g_j = hstack((reshape(grad_hid,(1,n_in*n_hid)), reshape(grad_out,(1,n_hid*n_out))))[0]
		g_j = array(g_j[0,:])[0]
		g += g_j/n_pat
	return -g

##########################
#     the good stuff     #
##########################

from scipy.optimize import fmin_cg

xor_in = matrix([	[0.0, 0.0],
					[0.0, 1.0],
					[1.0, 0.0],
					[1.0, 1.0]	])

xor_out = matrix([	[0.0],
					[1.0],
					[1.0],
					[0.0]	])

n_in = shape(xor_in)[1]
n_hid = 2
n_out = shape(xor_out)[1]
n_pats = shape(xor_in)[0]
wgt_in = rand(n_in,n_hid)*0.4 - 0.2
wgt_out = rand(n_hid,n_out)*0.4 - 0.2

w0 = hstack((reshape(wgt_in,(1,n_in*n_hid)), reshape(wgt_out,(1,n_hid*n_out))))[0]
params = [n_in, n_hid, n_out, xor_in, xor_out]

# optimize using conjugate gradient descent
w,f,fn,gn,warnflag,allvecs = fmin_cg(f, w0, fprime=fd, args=(params,), full_output=1, retall=1)

# print net performance
net_forward(w,params)












