# feedforward neural network
# input(2+bias) -> hidden(2+bias) -> output(1)
# trained on the XOR problem
# weights optimized using scipy.optimize.fmin_cg with
# gradients computed using backpropagation
# Paul Gribble, November 2012
# paul [at] gribblelab [dot] org

# ipython --pylab

import time
random.seed(int(time.time()))

def tansig(x):
	""" sigmoid activation function """
	return tanh(x)

def dtansig(x):
	""" derivative of sigmoid function """
	return 1.0 - (multiply(x,x)) # element-wise multiplication

def pack_weights(w_hid, b_hid, w_out, b_out, params):
	""" pack weight matrices into a single vector """
	n_in, n_hid, n_out = params[0], params[1], params[2]
	g_j = hstack((reshape(w_hid,(1,n_in*n_hid)), 
		      reshape(b_hid,(1,n_hid)),
		      reshape(w_out,(1,n_hid*n_out)),
		      reshape(b_out,(1,n_out))))[0]
	g_j = array(g_j[0,:])[0]
	return g_j

def unpack_weights(x, params):
	""" unpack weights from single vector into weight matrices """
	n_in, n_hid, n_out = params[0], params[1], params[2]
	pat_in, pat_out = params[3], params[4]
	n_pat = shape(pat_in)[0]
	i1,i2 = 0,n_in*n_hid
	w_hid = reshape(x[i1:i2], (n_in,n_hid))
	i1,i2 = i2,i2+n_hid
	b_hid = reshape(x[i1:i2],(1,n_hid))
	i1,i2 = i2,i2+(n_hid*n_out)
	w_out = reshape(x[i1:i2], (n_hid,n_out))
	i1,i2 = i2,i2+n_out
	b_out = reshape(x[i1:i2],(1,n_out))
	return w_hid, b_hid, w_out, b_out

def net_forward(x, params):
	""" propagate inputs through the network and return outputs """
	w_hid,b_hid,w_out,b_out = unpack_weights(x, params)
	pat_in = params[3]
	return tansig((tansig((pat_in * w_hid) + b_hid) * w_out) + b_out)

def f(x,params):
	""" returns the cost (SSE) of a given weight vector """
	pat_out = params[4]
	return sum(square(net_forward(x,params) - pat_out))

def fd(x,params):
	""" returns the gradients (dW/dE) for the weight vector """
	n_in, n_hid, n_out = params[0], params[1], params[2]
	pat_in, pat_out = params[3], params[4]
	w_hid,b_hid,w_out,b_out = unpack_weights(x, params)
	act_hid = tansig( (pat_in * w_hid) + b_hid )
	act_out = tansig( (act_hid * w_out) + b_out )
	err_out = act_out - pat_out
	deltas_out = multiply(dtansig(act_out), err_out)	
	err_hid = deltas_out * transpose(w_out)
	deltas_hid = multiply(dtansig(act_hid), err_hid)
	grad_w_out = transpose(act_hid)*deltas_out
	grad_b_out = sum(deltas_out,0)
	grad_w_hid = transpose(pat_in)*deltas_hid
	grad_b_hid = sum(deltas_hid,0)
	return pack_weights(grad_w_hid, grad_b_hid, grad_w_out, grad_b_out, params)

############################
#   train on XOR mapping   #
############################

from scipy.optimize import fmin_cg

xor_in = matrix([[0.0, 0.0],
		 [0.0, 1.0],
		 [1.0, 0.0],
		 [1.0, 1.0]	])

xor_out = matrix([[0.0],
		  [1.0],
		  [1.0],
		  [0.0]	])

# network parameters
n_in = shape(xor_in)[1]
n_hid = 2
n_out = shape(xor_out)[1]
params = [n_in, n_hid, n_out, xor_in, xor_out]

# initialize weights to small random values
nw = n_in*n_hid + n_hid + n_hid*n_out + n_out
w0 = rand(nw)*0.2 - 0.1

# optimize using conjugate gradient descent
out = fmin_cg(f, w0, fprime=fd, args=(params,),
	      full_output=True, retall=True, disp=True)
# unpack optimizer outputs
wopt,fopt,func_calls,grad_calls,warnflag,allvecs = out

# print net performance on optimal weights wopt
net_out = net_forward(wopt,params)
print net_out.round(3)

