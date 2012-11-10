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

def tansig(x):
	""" sigmoid activation function """
	return tanh(x)

def dtansig(x):
	""" derivative of sigmoid function """
	return 1.0 - (multiply(x,x))						 # element-wise multiplication

def net_forward(x, params):
	""" propagate inputs through the network and return outputs """
	n_in, n_hid, n_out = params[0], params[1], params[2] # unpack parameters from params list
	pat_in, pat_out = params[3], params[4]
	n_pat = shape(pat_in)[0]
	wgt_hid = reshape(x[0:n_in*n_hid], (n_in,n_hid))	 # weights as matrices
	wgt_out = reshape(x[n_in*n_hid:], (n_hid,n_out))
	return tansig(tansig(pat_in * wgt_hid) * wgt_out)	 # return network output

def f(x,params):
	""" returns the cost (SSE) of a given weight vector """
	pat_out = params[4]
	return sum(square(net_forward(x,params) - pat_out))

def fd(x,params):
	""" returns the gradients (dW/dE) for the weight vector """
	n_in, n_hid, n_out = params[0], params[1], params[2] # unpack parameters from params list
	pat_in, pat_out = params[3], params[4]
	n_pat = shape(pat_in)[0]
	wgt_hid = reshape(x[0:n_in*n_hid], (n_in,n_hid))	 # 
	wgt_out = reshape(x[n_in*n_hid:], (n_hid,n_out))
	act_hid = tansig( pat_in * wgt_hid )				 # unit activations
	act_out = tansig( act_hid * wgt_out )
	err_out = act_out - pat_out							 # output errors
	deltas_out = multiply(dtansig(act_out), err_out)     # output deltas
	err_hid = deltas_out * transpose(wgt_out)            # hidden errors
	deltas_hid = multiply(dtansig(act_hid), err_hid)     # hidden deltas
	grad_out = transpose(act_hid)*deltas_out			 # output gradients
	grad_hid = transpose(pat_in)*deltas_hid		     	 # hidden gradients
	# rearrange gradients as single vector
	g_j = hstack((reshape(grad_hid,(1,n_in*n_hid)), reshape(grad_out,(1,n_hid*n_out))))[0]
	g_j = array(g_j[0,:])[0]
	return g_j

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

# network parameters
n_in = shape(xor_in)[1]
n_hid = 2
n_out = shape(xor_out)[1]
n_pats = shape(xor_in)[0]
params = [n_in, n_hid, n_out, xor_in, xor_out]

# initialize weights to small random values
wgt_in = rand(n_in,n_hid)*0.2 - 0.1
wgt_out = rand(n_hid,n_out)*0.2 - 0.1
# pack weights into a single long array
w0 = hstack((reshape(wgt_in,(1,n_in*n_hid)), reshape(wgt_out,(1,n_hid*n_out))))[0]

# optimize using conjugate gradient descent
w,f,fn,gn,warnflag,allvecs = fmin_cg(f, w0, fprime=fd, args=(params,), full_output=1, retall=1)

# print net performance
net_out = net_forward(w,params)
print net_out.round(3)











